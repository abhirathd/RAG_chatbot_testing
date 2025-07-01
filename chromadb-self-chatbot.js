const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { glob } = require("glob");
const OpenAI = require("openai");
const { ChromaClient } = require("chromadb");
const { OpenAIEmbeddingFunction } = require("@chroma-core/openai");
require("dotenv").config();

// Using LangChain.js for document processing only
const { DirectoryLoader } = require("langchain/document_loaders/fs/directory");
const { TextLoader } = require("langchain/document_loaders/fs/text");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

// Configuration
const MODEL = "gpt-4o-mini";
const COLLECTION_NAME = "rag-documents";
const OPENAI_API_KEY = process.env.OPENAI_API_KEYY;
const CHROMA_HOST = process.env.CHROMA_HOST || "localhost";
const CHROMA_PORT = process.env.CHROMA_PORT || "8000";
// const CHROMA_URL = `http://${CHROMA_HOST}:${CHROMA_PORT}`;
const CHROMA_URL = `https://chroma-with-credentials-8y7a.onrender.com`;

const embedder = new OpenAIEmbeddingFunction({
  openai_api_key: OPENAI_API_KEY,
  openai_model: "text-embedding-3-small",
});

class ChromaRAGChatbot {
  constructor() {
    this.chromaClient = null;
    this.collection = null;
    this.openai = null;
    this.chatHistory = [];
    this.isInitialized = false;
    this.documentCount = 0;
  }

  async initialize() {
    try {
      console.log("Initializing ChromaDB RAG Chatbot...");
      console.log("=".repeat(50));

      // Initialize OpenAI client
      if (!OPENAI_API_KEY || OPENAI_API_KEY === "your-openai-api-key") {
        throw new Error("OPENAI_API_KEY environment variable is required!");
      }

      this.openai = new OpenAI({
        apiKey: OPENAI_API_KEY,
      });

      // Initialize ChromaDB connection
      await this.initializeChroma();

      // Check if collection exists and has data
      const collectionExists = await this.checkCollectionExists();

      if (!collectionExists) {
        console.log("Creating new ChromaDB collection...");
        await this.createCollection();
      } else {
        console.log("Using existing ChromaDB collection.");
        await this.loadExistingCollection();
      }

      // Check if we need to populate the collection
      this.documentCount = await this.collection.count();

      if (this.documentCount === 0) {
        console.log("Collection is empty. Creating knowledge base...");
        await this.createKnowledgeBase();
      }

      this.isInitialized = true;

      console.log("=".repeat(50));
      console.log("ChromaDB RAG Chatbot initialized successfully!");
      console.log(`Knowledge base contains ${this.documentCount} documents.`);
      console.log("You can now ask questions about your documents.");
      console.log("Commands:");
      console.log('  - "rebuild" - Rebuild the knowledge base from documents');
      console.log('  - "status" - Show current status');
      console.log('  - "exit" or "quit" - End the conversation');
      console.log("=".repeat(50));
    } catch (error) {
      console.error("Failed to initialize chatbot:", error.message);
      throw error;
    }
  }

  async initializeChroma() {
    const initStartTime = Date.now();
    console.log(
      `⏱️ [${new Date().toISOString()}] Connecting to ChromaDB at ${CHROMA_URL}...`
    );

    try {
      // Create ChromaDB client
      this.chromaClient = new ChromaClient({
        host: "chroma-with-credentials-8y7a.onrender.com",
        port: 443, // Use 443 for HTTPS, 80 for HTTP
        ssl: true, // Set to true for HTTPS URLs
      });

      // Test connection with heartbeat
      await this.chromaClient.heartbeat();
      console.log(`✅ Connected to ChromaDB successfully`);

      const connectTime = Date.now() - initStartTime;
      console.log(`⏱️ ChromaDB connection: ${connectTime}ms`);
    } catch (error) {
      const connectTime = Date.now() - initStartTime;
      console.error(
        `❌ ChromaDB connection failed after ${connectTime}ms:`,
        error.message
      );
      console.error(`Make sure ChromaDB is running at ${CHROMA_URL}`);
      console.error(
        "Start ChromaDB with: docker run -p 8000:8000 chromadb/chroma"
      );
      throw error;
    }
  }

  async checkCollectionExists() {
    try {
      const collections = await this.chromaClient.listCollections();
      console.log("collections", collections);
      return collections.some(
        (collection) => collection.name === COLLECTION_NAME
      );
    } catch (error) {
      console.error("Error checking collection existence:", error.message);
      return false;
    }
  }

  async createCollection() {
    try {
      console.log(`Creating ChromaDB collection: ${COLLECTION_NAME}...`);

      this.collection = await this.chromaClient.createCollection({
        name: COLLECTION_NAME,
        metadata: {
          description: "RAG chatbot document embeddings",
          created_at: new Date().toISOString(),
        },
        embeddingFunction: embedder, // We'll provide our own embeddings
      });

      console.log("✅ ChromaDB collection created successfully");
    } catch (error) {
      console.error("Error creating collection:", error.message);
      throw error;
    }
  }

  async loadExistingCollection() {
    try {
      this.collection = await this.chromaClient.getCollection({
        name: COLLECTION_NAME,
      });

      // Get document count
      this.documentCount = await this.collection.count();

      console.log(
        `✅ Loaded existing collection with ${this.documentCount} documents`
      );
    } catch (error) {
      console.error("Error loading existing collection:", error.message);
      throw error;
    }
  }

  async createKnowledgeBase() {
    try {
      const documents = await this.loadDocuments();
      const chunks = await this.createTextChunks(documents);
      await this.createVectorStore(chunks);
    } catch (error) {
      console.error("Error creating knowledge base:", error.message);
      throw error;
    }
  }

  async loadDocuments() {
    console.log("Loading documents from knowledge-base...");

    try {
      // Check if knowledge-base directory exists
      if (!fs.existsSync("knowledge-base")) {
        console.log(
          "⚠️ knowledge-base directory not found. Creating example..."
        );
        this.createExampleKnowledgeBase();
        return [];
      }

      const folders = await glob("knowledge-base/*", { onlyDirectories: true });
      const documents = [];

      if (folders.length === 0) {
        console.log("⚠️ No folders found in knowledge-base directory");
        return [];
      }

      for (const folder of folders) {
        const docType = path.basename(folder);
        console.log(`Processing folder: ${docType}`);

        try {
          const loader = new DirectoryLoader(folder, {
            ".md": (path) => new TextLoader(path),
            ".txt": (path) => new TextLoader(path),
          });

          const folderDocs = await loader.load();

          // Add document type metadata
          folderDocs.forEach((doc) => {
            doc.metadata.doc_type = docType;
            doc.metadata.source = path.relative(
              "knowledge-base",
              doc.metadata.source
            );
            documents.push(doc);
          });

          console.log(
            `  - Loaded ${folderDocs.length} documents from ${docType}`
          );
        } catch (error) {
          console.error(`  - Error loading from ${docType}:`, error.message);
        }
      }

      console.log(`📚 Total documents loaded: ${documents.length}`);
      return documents;
    } catch (error) {
      console.error("Error loading documents:", error.message);
      throw error;
    }
  }

  createExampleKnowledgeBase() {
    const exampleDir = "knowledge-base/example";
    fs.mkdirSync(exampleDir, { recursive: true });

    const exampleContent = `# Example Document

This is an example document for the RAG chatbot.

## Features
- Supports markdown files
- Can handle multiple document types
- Provides contextual answers

## Usage
Place your documents in the knowledge-base directory, organized by folders.
`;

    fs.writeFileSync(path.join(exampleDir, "example.md"), exampleContent);
    console.log(
      "📝 Created example knowledge base. Add your documents to knowledge-base/ directory."
    );
  }

  async createTextChunks(documents) {
    if (documents.length === 0) {
      console.log("⚠️ No documents to process");
      return [];
    }

    console.log("Creating text chunks...");

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const chunks = await textSplitter.splitDocuments(documents);
    console.log(`📄 Created ${chunks.length} text chunks`);

    // Display document types found
    const docTypes = [
      ...new Set(chunks.map((chunk) => chunk.metadata.doc_type)),
    ];
    console.log(`📋 Document types found: ${docTypes.join(", ")}`);

    return chunks;
  }

  async createVectorStore(chunks) {
    if (chunks.length === 0) {
      console.log("⚠️ No chunks to create vector store");
      return;
    }

    console.log("Creating embeddings and vector store...");
    const embeddingStartTime = Date.now();

    try {
      // Process chunks in batches to avoid memory issues
      const batchSize = 100;

      for (let i = 0; i < chunks.length; i += batchSize) {
        const batch = chunks.slice(i, i + batchSize);
        console.log(
          `📄 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(
            chunks.length / batchSize
          )}...`
        );

        const ids = [];
        const embeddings = [];
        const documents = [];
        const metadatas = [];

        for (let j = 0; j < batch.length; j++) {
          const chunk = batch[j];
          const globalIndex = i + j;

          // Get embedding from OpenAI
          const embedding = await this.getEmbedding(chunk.pageContent);

          ids.push(`doc_${globalIndex}`);
          embeddings.push(embedding);
          documents.push(chunk.pageContent);
          metadatas.push({
            doc_type: chunk.metadata.doc_type,
            source: chunk.metadata.source,
            chunk_index: globalIndex.toString(),
          });
        }

        // Add batch to ChromaDB
        await this.collection.add({
          ids: ids,
          embeddings: embeddings,
          documents: documents,
          metadatas: metadatas,
        });

        // Small delay to be gentle on the system
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      this.documentCount = chunks.length;

      const totalTime = Date.now() - embeddingStartTime;
      console.log(`⏱️ Vector store creation: ${totalTime}ms`);
      console.log("✅ Vector store created successfully");
    } catch (error) {
      console.error("❌ Error creating vector store:", error.message);
      throw error;
    }
  }

  async getEmbedding(text) {
    try {
      const response = await this.openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error("❌ Embedding failed:", error.message);
      throw error;
    }
  }

  async retrieveRelevantDocuments(query, limit = 4) {
    if (!this.collection) {
      console.log("⚠️ No collection available for search");
      return [];
    }

    const searchStartTime = Date.now();
    console.log(`🔍 Searching for relevant documents...`);

    try {
      // Get query embedding
      const queryEmbedding = await this.getEmbedding(query);

      // Perform vector search
      const results = await this.collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: limit,
        include: ["documents", "metadatas", "distances"],
      });

      const searchTime = Date.now() - searchStartTime;
      console.log(`⏱️ Document search: ${searchTime}ms`);
      console.log(
        `📚 Found ${results.documents[0]?.length || 0} relevant documents`
      );

      // Format results to match expected structure
      const formattedResults = [];
      if (results.documents[0]) {
        for (let i = 0; i < results.documents[0].length; i++) {
          formattedResults.push({
            document: results.documents[0][i],
            metadata: results.metadatas[0][i],
            distance: results.distances[0][i],
          });
        }
      }

      return formattedResults;
    } catch (error) {
      console.error(`❌ Search failed:`, error.message);
      return [];
    }
  }

  formatChatHistory() {
    return this.chatHistory
      .map((msg) => {
        return `${msg.role === "user" ? "Human" : "Assistant"}: ${msg.content}`;
      })
      .join("\n");
  }

  async chat(message) {
    try {
      // Handle special commands
      if (message.toLowerCase() === "rebuild") {
        console.log("\n🔄 Rebuilding knowledge base...");
        await this.rebuildKnowledgeBase();
        return "Knowledge base rebuilt successfully!";
      }

      if (message.toLowerCase() === "status") {
        return this.getStatus();
      }

      // Retrieve relevant documents
      const relevantDocs = await this.retrieveRelevantDocuments(message);

      // Format context from retrieved documents
      const context = relevantDocs
        .map((result, index) => {
          const metadata = result.metadata;
          return `Document ${index + 1} (${metadata.doc_type}) - Source: ${
            metadata.source
          }:\n${result.document}`;
        })
        .join("\n\n");

      // Format chat history
      const historyContext =
        this.chatHistory.length > 0
          ? `Previous conversation:\n${this.formatChatHistory()}\n\n`
          : "";

      // Create the system prompt
      const systemPrompt = `You are a helpful AI assistant that answers questions based on the provided context documents. Use the information from the documents to provide accurate and helpful responses. If the information isn't available in the context, say so clearly.

${historyContext}Context Documents:
${context}

Please answer the following question based on the context provided above.`;

      const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ];

      // Store user message in history
      this.chatHistory.push({ role: "user", content: message });

      // Create streaming chat completion
      const stream = await this.openai.chat.completions.create({
        model: MODEL,
        messages: messages,
        temperature: 0.7,
        stream: true,
      });

      let fullResponse = "";

      // Process the stream
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          process.stdout.write(content);
          fullResponse += content;
        }
      }

      // Store assistant response in history
      this.chatHistory.push({ role: "assistant", content: fullResponse });

      // Keep chat history manageable (last 20 messages)
      if (this.chatHistory.length > 20) {
        this.chatHistory = this.chatHistory.slice(-20);
      }

      return fullResponse;
    } catch (error) {
      console.error("Error during chat:", error.message);
      return "Sorry, I encountered an error processing your question. Please try again.";
    }
  }

  async rebuildKnowledgeBase() {
    try {
      // Delete the existing collection
      try {
        await this.chromaClient.deleteCollection({
          name: COLLECTION_NAME,
        });
        console.log("🗑️ Deleted existing collection");
      } catch (error) {
        // Collection might not exist, that's OK
        console.log("No existing collection to delete");
      }

      // Recreate the collection
      await this.createCollection();

      // Recreate the knowledge base
      await this.createKnowledgeBase();
      console.log("✅ Knowledge base rebuilt successfully");
    } catch (error) {
      console.error("❌ Error rebuilding knowledge base:", error.message);
      throw error;
    }
  }

  async getStatus() {
    let collectionInfo = "Not available";
    if (this.collection) {
      try {
        const count = await this.collection.count();
        collectionInfo = `${count} documents`;
      } catch (error) {
        collectionInfo = "Error fetching count";
      }
    }

    return `📊 System Status:
- ChromaDB: ${
      this.chromaClient ? "✅ Connected" : "❌ Not connected"
    } (${CHROMA_URL})
- Collection: ${this.collection ? "✅ Available" : "❌ Not available"}
- Documents: ${collectionInfo}
- Chat History: ${this.chatHistory.length} messages
- Model: ${MODEL}`;
  }

  async startTerminalInterface() {
    if (!this.isInitialized) {
      throw new Error("Chatbot not initialized. Call initialize() first.");
    }

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const askQuestion = () => {
      rl.question("\n🤖 You: ", async (input) => {
        const message = input.trim();

        if (
          message.toLowerCase() === "exit" ||
          message.toLowerCase() === "quit"
        ) {
          console.log(
            "\n👋 Goodbye! Thanks for using the ChromaDB RAG Chatbot."
          );
          rl.close();
          return;
        }

        if (message === "") {
          askQuestion();
          return;
        }

        console.log("\n💬 Bot: ");
        await this.chat(message);
        console.log("\n"); // Add extra line after streaming response

        askQuestion();
      });
    };

    askQuestion();
  }
}

// Main execution
async function main() {
  try {
    const chatbot = new ChromaRAGChatbot();
    await chatbot.initialize();
    await chatbot.startTerminalInterface();
  } catch (error) {
    console.error("Fatal error:", error.message);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.log("\n\n👋 Received interrupt signal. Shutting down gracefully...");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log(
    "\n\n👋 Received termination signal. Shutting down gracefully..."
  );
  process.exit(0);
});

// Export for module use
module.exports = { ChromaRAGChatbot };

// Start the application
if (require.main === module) {
  main();
}
