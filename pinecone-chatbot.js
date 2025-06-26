const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { glob } = require('glob');
const OpenAI = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
require('dotenv').config();

// Using LangChain.js for document processing only
const { DirectoryLoader } = require('langchain/document_loaders/fs/directory');
const { TextLoader } = require('langchain/document_loaders/fs/text');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

// Configuration
const MODEL = "gpt-4o";
const INDEX_NAME = "rag-documents";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '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';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || 'your-pinecone-api-key';
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || 'us-east-1-aws';

class PineconeRAGChatbot {
    constructor() {
        this.pinecone = null;
        this.index = null;
        this.openai = null;
        this.chatHistory = [];
        this.isInitialized = false;
        this.documentCount = 0;
    }

    async initialize() {
        try {
            console.log('Initializing Pinecone RAG Chatbot...');
            console.log('='.repeat(50));
            
            // Initialize OpenAI client
            if (!OPENAI_API_KEY || OPENAI_API_KEY === 'your-openai-api-key') {
                throw new Error('OPENAI_API_KEY environment variable is required!');
            }
            
            this.openai = new OpenAI({
                apiKey: OPENAI_API_KEY,
            });

            // Initialize Pinecone connection
            await this.initializePinecone();
            
            // Check if index exists and has data
            const indexExists = await this.checkIndexExists();
            
            if (!indexExists) {
                console.log('Creating new Pinecone index...');
                await this.createIndex();
            } else {
                console.log('Using existing Pinecone index.');
                await this.loadExistingIndex();
            }
            
            // Check if we need to populate the index
            const stats = await this.index.describeIndexStats();
            this.documentCount = stats.totalVectorCount || 0;
            
            if (this.documentCount === 0) {
                console.log('Index is empty. Creating knowledge base...');
                await this.createKnowledgeBase();
            }
            
            this.isInitialized = true;
            
            console.log('='.repeat(50));
            console.log('Pinecone RAG Chatbot initialized successfully!');
            console.log(`Knowledge base contains ${this.documentCount} documents.`);
            console.log('You can now ask questions about your documents.');
            console.log('Commands:');
            console.log('  - "rebuild" - Rebuild the knowledge base from documents');
            console.log('  - "status" - Show current status');
            console.log('  - "exit" or "quit" - End the conversation');
            console.log('='.repeat(50));
            
        } catch (error) {
            console.error('Failed to initialize chatbot:', error.message);
            throw error;
        }
    }

    async initializePinecone() {
        const initStartTime = Date.now();
        console.log(`⏱️ [${new Date().toISOString()}] Connecting to Pinecone...`);
        
        try {
            if (!PINECONE_API_KEY || PINECONE_API_KEY === 'your-pinecone-api-key') {
                throw new Error('PINECONE_API_KEY environment variable is required!');
            }

            this.pinecone = new Pinecone({
                apiKey: PINECONE_API_KEY,
            });
            
            const connectTime = Date.now() - initStartTime;
            console.log(`⏱️ Pinecone connection: ${connectTime}ms`);
            console.log("✅ Connected to Pinecone successfully");
            
        } catch (error) {
            const connectTime = Date.now() - initStartTime;
            console.error(`❌ Pinecone connection failed after ${connectTime}ms:`, error.message);
            throw error;
        }
    }

    async checkIndexExists() {
        try {
            const indexList = await this.pinecone.listIndexes();
            return indexList.indexes?.some(index => index.name === INDEX_NAME) || false;
        } catch (error) {
            console.error('Error checking index existence:', error.message);
            return false;
        }
    }

    async createIndex() {
        try {
            console.log(`Creating Pinecone index: ${INDEX_NAME}...`);
            
            await this.pinecone.createIndex({
                name: INDEX_NAME,
                dimension: 1536, // text-embedding-3-small dimension
                metric: 'cosine',
                spec: {
                    serverless: {
                        cloud: 'aws',
                        region: 'us-east-1'
                    }
                }
            });

            // Wait for index to be ready
            console.log('Waiting for index to be ready...');
            let isReady = false;
            while (!isReady) {
                const indexDescription = await this.pinecone.describeIndex(INDEX_NAME);
                isReady = indexDescription.status?.ready || false;
                if (!isReady) {
                    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
                }
            }

            this.index = this.pinecone.index(INDEX_NAME);
            console.log('✅ Pinecone index created and ready');
            
        } catch (error) {
            console.error('Error creating index:', error.message);
            throw error;
        }
    }

    async loadExistingIndex() {
        try {
            this.index = this.pinecone.index(INDEX_NAME);
            
            // Get index stats
            const stats = await this.index.describeIndexStats();
            this.documentCount = stats.totalVectorCount || 0;
            
            console.log(`✅ Loaded existing index with ${this.documentCount} documents`);
        } catch (error) {
            console.error('Error loading existing index:', error.message);
            throw error;
        }
    }

    async createKnowledgeBase() {
        try {
            const documents = await this.loadDocuments();
            const chunks = await this.createTextChunks(documents);
            await this.createVectorStore(chunks);
        } catch (error) {
            console.error('Error creating knowledge base:', error.message);
            throw error;
        }
    }

    async loadDocuments() {
        console.log('Loading documents from knowledge-base...');
        
        try {
            // Check if knowledge-base directory exists
            if (!fs.existsSync('knowledge-base')) {
                console.log('⚠️ knowledge-base directory not found. Creating example...');
                this.createExampleKnowledgeBase();
                return [];
            }

            const folders = await glob('knowledge-base/*', { onlyDirectories: true });
            const documents = [];

            if (folders.length === 0) {
                console.log('⚠️ No folders found in knowledge-base directory');
                return [];
            }

            for (const folder of folders) {
                const docType = path.basename(folder);
                console.log(`Processing folder: ${docType}`);
                
                try {
                    const loader = new DirectoryLoader(folder, {
                        '.md': (path) => new TextLoader(path),
                        '.txt': (path) => new TextLoader(path),
                    });
                    
                    const folderDocs = await loader.load();
                    
                    // Add document type metadata
                    folderDocs.forEach(doc => {
                        doc.metadata.doc_type = docType;
                        doc.metadata.source = path.relative('knowledge-base', doc.metadata.source);
                        documents.push(doc);
                    });
                    
                    console.log(`  - Loaded ${folderDocs.length} documents from ${docType}`);
                } catch (error) {
                    console.error(`  - Error loading from ${docType}:`, error.message);
                }
            }

            console.log(`📚 Total documents loaded: ${documents.length}`);
            return documents;
        } catch (error) {
            console.error('Error loading documents:', error.message);
            throw error;
        }
    }

    createExampleKnowledgeBase() {
        const exampleDir = 'knowledge-base/example';
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
        
        fs.writeFileSync(path.join(exampleDir, 'example.md'), exampleContent);
        console.log('📝 Created example knowledge base. Add your documents to knowledge-base/ directory.');
    }

    async createTextChunks(documents) {
        if (documents.length === 0) {
            console.log('⚠️ No documents to process');
            return [];
        }

        console.log('Creating text chunks...');
        
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunks = await textSplitter.splitDocuments(documents);
        console.log(`📄 Created ${chunks.length} text chunks`);

        // Display document types found
        const docTypes = [...new Set(chunks.map(chunk => chunk.metadata.doc_type))];
        console.log(`📋 Document types found: ${docTypes.join(', ')}`);

        return chunks;
    }

    async createVectorStore(chunks) {
        if (chunks.length === 0) {
            console.log('⚠️ No chunks to create vector store');
            return;
        }

        console.log('Creating embeddings and vector store...');
        const embeddingStartTime = Date.now();

        try {
            // Process chunks in batches to avoid rate limits
            const batchSize = 100;
            const vectors = [];
            
            for (let i = 0; i < chunks.length; i += batchSize) {
                const batch = chunks.slice(i, i + batchSize);
                console.log(`📄 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(chunks.length/batchSize)}...`);
                
                for (let j = 0; j < batch.length; j++) {
                    const chunk = batch[j];
                    const globalIndex = i + j;
                    
                    // Get embedding from OpenAI
                    const embedding = await this.getEmbedding(chunk.pageContent);
                    
                    vectors.push({
                        id: `doc_${globalIndex}`,
                        values: embedding,
                        metadata: {
                            text: chunk.pageContent,
                            doc_type: chunk.metadata.doc_type,
                            source: chunk.metadata.source
                        }
                    });
                }
                
                // Upsert batch to Pinecone
                if (vectors.length > 0) {
                    await this.index.upsert(vectors.splice(0, vectors.length));
                }
                
                // Small delay to respect rate limits
                await new Promise(resolve => setTimeout(resolve, 100));
            }

            this.documentCount = chunks.length;
            
            const totalTime = Date.now() - embeddingStartTime;
            console.log(`⏱️ Vector store creation: ${totalTime}ms`);
            console.log('✅ Vector store created successfully');
            
        } catch (error) {
            console.error('❌ Error creating vector store:', error.message);
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
            console.error('❌ Embedding failed:', error.message);
            throw error;
        }
    }

    async retrieveRelevantDocuments(query, limit = 4) {
        if (!this.index) {
            console.log("⚠️ No index available for search");
            return [];
        }

        const searchStartTime = Date.now();
        console.log(`🔍 Searching for relevant documents...`);

        try {
            // Get query embedding
            const queryEmbedding = await this.getEmbedding(query);
            
            // Perform vector search
            const queryResponse = await this.index.query({
                vector: queryEmbedding,
                topK: limit,
                includeMetadata: true
            });
            
            const searchTime = Date.now() - searchStartTime;
            console.log(`⏱️ Document search: ${searchTime}ms`);
            console.log(`📚 Found ${queryResponse.matches?.length || 0} relevant documents`);
            
            return queryResponse.matches || [];
        } catch (error) {
            console.error(`❌ Search failed:`, error.message);
            return [];
        }
    }

    formatChatHistory() {
        return this.chatHistory.map(msg => {
            return `${msg.role === 'user' ? 'Human' : 'Assistant'}: ${msg.content}`;
        }).join('\n');
    }

    async chat(message) {
        try {
            // Handle special commands
            if (message.toLowerCase() === 'rebuild') {
                console.log('\n🔄 Rebuilding knowledge base...');
                await this.rebuildKnowledgeBase();
                return 'Knowledge base rebuilt successfully!';
            }
            
            if (message.toLowerCase() === 'status') {
                return this.getStatus();
            }

            // Retrieve relevant documents
            const relevantDocs = await this.retrieveRelevantDocuments(message);
            
            // Format context from retrieved documents
            const context = relevantDocs.map((match, index) => {
                const metadata = match.metadata;
                return `Document ${index + 1} (${metadata.doc_type}) - Source: ${metadata.source}:\n${metadata.text}`;
            }).join('\n\n');

            // Format chat history
            const historyContext = this.chatHistory.length > 0 ? 
                `Previous conversation:\n${this.formatChatHistory()}\n\n` : '';

            // Create the system prompt
            const systemPrompt = `You are a helpful AI assistant that answers questions based on the provided context documents. Use the information from the documents to provide accurate and helpful responses. If the information isn't available in the context, say so clearly.

${historyContext}Context Documents:
${context}

Please answer the following question based on the context provided above.`;

            const messages = [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: message }
            ];

            // Store user message in history
            this.chatHistory.push({ role: 'user', content: message });

            // Create streaming chat completion
            const stream = await this.openai.chat.completions.create({
                model: MODEL,
                messages: messages,
                temperature: 0.7,
                stream: true,
            });

            let fullResponse = '';
            
            // Process the stream
            for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content || '';
                if (content) {
                    process.stdout.write(content);
                    fullResponse += content;
                }
            }

            // Store assistant response in history
            this.chatHistory.push({ role: 'assistant', content: fullResponse });

            // Keep chat history manageable (last 20 messages)
            if (this.chatHistory.length > 20) {
                this.chatHistory = this.chatHistory.slice(-20);
            }

            return fullResponse;

        } catch (error) {
            console.error('Error during chat:', error.message);
            return 'Sorry, I encountered an error processing your question. Please try again.';
        }
    }

    async rebuildKnowledgeBase() {
        try {
            // Delete all vectors from the index
            await this.index.deleteAll();
            console.log('🗑️ Cleared existing vectors from index');
            
            // Recreate the knowledge base
            await this.createKnowledgeBase();
            console.log('✅ Knowledge base rebuilt successfully');
            
        } catch (error) {
            console.error('❌ Error rebuilding knowledge base:', error.message);
            throw error;
        }
    }

    async getStatus() {
        let indexStats = 'Not available';
        if (this.index) {
            try {
                const stats = await this.index.describeIndexStats();
                indexStats = `${stats.totalVectorCount || 0} vectors`;
            } catch (error) {
                indexStats = 'Error fetching stats';
            }
        }

        return `📊 System Status:
- Pinecone: ${this.pinecone ? '✅ Connected' : '❌ Not connected'}
- Index: ${this.index ? '✅ Available' : '❌ Not available'}
- Documents: ${indexStats}
- Chat History: ${this.chatHistory.length} messages
- Model: ${MODEL}`;
    }

    async startTerminalInterface() {
        if (!this.isInitialized) {
            throw new Error('Chatbot not initialized. Call initialize() first.');
        }

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        const askQuestion = () => {
            rl.question('\n🤖 You: ', async (input) => {
                const message = input.trim();
                
                if (message.toLowerCase() === 'exit' || message.toLowerCase() === 'quit') {
                    console.log('\n👋 Goodbye! Thanks for using the Pinecone RAG Chatbot.');
                    rl.close();
                    return;
                }

                if (message === '') {
                    askQuestion();
                    return;
                }

                console.log('\n💬 Bot: ');
                await this.chat(message);
                console.log('\n'); // Add extra line after streaming response
                
                askQuestion();
            });
        };

        askQuestion();
    }
}

// Main execution
async function main() {
    try {
        const chatbot = new PineconeRAGChatbot();
        await chatbot.initialize();
        await chatbot.startTerminalInterface();
    } catch (error) {
        console.error('Fatal error:', error.message);
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\n👋 Received interrupt signal. Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n\n👋 Received termination signal. Shutting down gracefully...');
    process.exit(0);
});

// Export for module use
module.exports = { PineconeRAGChatbot };

// Start the application
if (require.main === module) {
    main();
}