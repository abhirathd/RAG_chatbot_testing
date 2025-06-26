const { ChromaClient } = require("chromadb-client");
const OpenAI = require("openai");
const dotenv = require("dotenv");
dotenv.config();

// Initialize clients
const chromaClient = new ChromaClient({ path: "http://localhost:8000" });
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

class RAGSystem {
  constructor(collectionName = "documents") {
    this.collectionName = collectionName;
    this.collection = null;
  }

  // Initialize or get existing collection
  async initializeCollection() {
    try {
      this.collection = await chromaClient.getOrCreateCollection({
        name: this.collectionName,
        metadata: { "hnsw:space": "cosine" }
      });
      console.log(`Collection "${this.collectionName}" initialized`);
    } catch (error) {
      console.error("Error initializing collection:", error);
      throw error;
    }
  }

  // Generate embeddings using OpenAI
  async generateEmbedding(text) {
    try {
      const response = await openai.embeddings.create({
        model: "text-embedding-3-small", // or "text-embedding-3-large"
        input: text,
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error("Error generating embedding:", error);
      throw error;
    }
  }

  // Add documents to ChromaDB
  async addDocuments(documents) {
    if (!this.collection) {
      await this.initializeCollection();
    }

    try {
      const embeddings = [];
      const ids = [];
      const metadatas = [];
      const docs = [];

      for (let i = 0; i < documents.length; i++) {
        const doc = documents[i];
        const embedding = await this.generateEmbedding(doc.content);
        
        embeddings.push(embedding);
        ids.push(doc.id || `doc_${Date.now()}_${i}`);
        metadatas.push(doc.metadata || {});
        docs.push(doc.content);
      }

      await this.collection.add({
        embeddings: embeddings,
        documents: docs,
        metadatas: metadatas,
        ids: ids
      });

      console.log(`Added ${documents.length} documents to collection`);
    } catch (error) {
      console.error("Error adding documents:", error);
      throw error;
    }
  }

  // Search for similar documents
  async searchSimilar(query, nResults = 5) {
    if (!this.collection) {
      await this.initializeCollection();
    }

    try {
      const queryEmbedding = await this.generateEmbedding(query);
      
      const results = await this.collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: nResults,
        include: ["documents", "metadatas", "distances"]
      });

      return results;
    } catch (error) {
      console.error("Error searching documents:", error);
      throw error;
    }
  }

  // Generate response using retrieved context
  async generateResponse(query, maxTokens = 1000) {
    try {
      // Retrieve relevant documents
      const searchResults = await this.searchSimilar(query);
      
      // Combine retrieved documents as context
      const context = searchResults.documents[0].join("\n\n");
      
      // Create prompt with context
      const prompt = `Context information is below:
${context}

Given the context information above, please answer the following question:
${query}

If the answer cannot be found in the context, please say so.`;

      // Generate response using OpenAI
      const response = await openai.chat.completions.create({
        model: "gpt-4", // or "gpt-3.5-turbo"
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant that answers questions based on the provided context."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: maxTokens,
        temperature: 0.7
      });

      return {
        answer: response.choices[0].message.content,
        sources: searchResults.documents[0],
        distances: searchResults.distances[0]
      };
    } catch (error) {
      console.error("Error generating response:", error);
      throw error;
    }
  }
}

// Usage example
async function main() {
  const rag = new RAGSystem("my_knowledge_base");

  // Example documents to add
  const documents = [
    {
      id: "doc1",
      content: "Artificial Intelligence is a field of computer science that aims to create intelligent machines.",
      metadata: { source: "AI textbook", chapter: 1 }
    },
    {
      id: "doc2", 
      content: "Machine Learning is a subset of AI that focuses on algorithms that can learn from data.",
      metadata: { source: "ML guide", topic: "fundamentals" }
    },
    {
      id: "doc3",
      content: "Neural networks are computing systems inspired by biological neural networks.",
      metadata: { source: "Deep Learning book", chapter: 2 }
    }
  ];

  try {
    // Add documents to the knowledge base
    await rag.addDocuments(documents);

    // Ask a question
    const query = "What is machine learning?";
    const result = await rag.generateResponse(query);

    console.log("Question:", query);
    console.log("Answer:", result.answer);
    console.log("Sources used:", result.sources.length);
    
  } catch (error) {
    console.error("Error in main:", error);
  }
}

// Run the example
main();