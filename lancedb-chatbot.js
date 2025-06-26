const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { glob } = require('glob');
const OpenAI = require('openai');
const lancedb = require("@lancedb/lancedb");
require('dotenv').config();

// Using LangChain.js for document processing only
const { DirectoryLoader } = require('langchain/document_loaders/fs/directory');
const { TextLoader } = require('langchain/document_loaders/fs/text');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

// Configuration
const MODEL = "gpt-4o-mini";
const TABLE_NAME = "documents";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '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';
const LANCEDB_URI = process.env.LANCEDB_URI || "db://default-8zm8d7"; // Replace with your URI
const LANCEDB_API_KEY = process.env.LANCEDB_API_KEY || "sk_EBHPQIN2SJGVTJIPPPG2UTUDY7PSLBNNELNUKTEIFQ2J6TDDIYEA===="; // Replace with your API key

class LanceDBRAGChatbot {
    constructor() {
        this.db = null;
        this.table = null;
        this.openai = null;
        this.chatHistory = [];
        this.isInitialized = false;
        this.documentCount = 0;
    }

    async initialize() {
        try {
            console.log('Initializing LanceDB RAG Chatbot...');
            console.log('='.repeat(50));
            
            // Initialize OpenAI client
            if (!OPENAI_API_KEY) {
                throw new Error('OPENAI_API_KEY environment variable is required!');
            }
            
            this.openai = new OpenAI({
                apiKey: OPENAI_API_KEY,
            });

            // Initialize LanceDB connection
            await this.initializeLanceDB();
            
            // Check if table exists and has data
            const tableExists = await this.checkTableExists();
            
            if (!tableExists) {
                console.log('No existing table found. Creating new knowledge base...');
                await this.createKnowledgeBase();
            } else {
                console.log('Using existing knowledge base.');
                await this.loadExistingTable();
            }
            
            this.isInitialized = true;
            
            console.log('='.repeat(50));
            console.log('LanceDB RAG Chatbot initialized successfully!');
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

    async initializeLanceDB() {
        const initStartTime = Date.now();
        console.log(`⏱️ [${new Date().toISOString()}] Connecting to LanceDB...`);
        
        try {
            this.db = await lancedb.connect({
                uri: LANCEDB_URI,
                apiKey: LANCEDB_API_KEY,
                region: "us-east-1",
                streams: true
            });
            
            const connectTime = Date.now() - initStartTime;
            console.log(`⏱️ LanceDB connection: ${connectTime}ms`);
            console.log("✅ Connected to LanceDB successfully");
            
        } catch (error) {
            const connectTime = Date.now() - initStartTime;
            console.error(`❌ LanceDB connection failed after ${connectTime}ms:`, error.message);
            throw error;
        }
    }

    async checkTableExists() {
        try {
            const tableNames = await this.db.tableNames();
            return tableNames.includes(TABLE_NAME);
        } catch (error) {
            console.error('Error checking table existence:', error.message);
            return false;
        }
    }

    async loadExistingTable() {
        try {
            this.table = await this.db.openTable(TABLE_NAME);
            
            // Get document count
            const countResult = await this.table.countRows();
            this.documentCount = countResult;
            
            console.log(`✅ Loaded existing table with ${this.documentCount} documents`);
        } catch (error) {
            console.error('Error loading existing table:', error.message);
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
            // Prepare data for LanceDB
            const data = [];
            
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i];
                console.log(`📄 Processing chunk ${i + 1}/${chunks.length}...`);
                
                // Get embedding from OpenAI
                const embedding = await this.getEmbedding(chunk.pageContent);
                
                data.push({
                    id: i,
                    text: chunk.pageContent,
                    doc_type: chunk.metadata.doc_type,
                    source: chunk.metadata.source,
                    vector: embedding
                });
            }

            // Create table
            console.log('💾 Creating LanceDB table...');
            this.table = await this.db.createTable(TABLE_NAME, data);
            this.documentCount = data.length;
            
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
        if (!this.table) {
            console.log("⚠️ No table available for search");
            return [];
        }

        const searchStartTime = Date.now();
        console.log(`🔍 Searching for relevant documents...`);

        try {
            // Get query embedding
            const queryEmbedding = await this.getEmbedding(query);
            
            // Perform vector search
            const results = await this.table
                .search(queryEmbedding)
                .limit(limit)
                .toArray();
            
            const searchTime = Date.now() - searchStartTime;
            console.log(`⏱️ Document search: ${searchTime}ms`);
            console.log(`📚 Found ${results.length} relevant documents`);
            
            return results;
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
            const context = relevantDocs.map((doc, index) => {
                return `Document ${index + 1} (${doc.doc_type}) - Source: ${doc.source}:\n${doc.text}`;
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
            // Drop existing table if it exists
            try {
                await this.db.dropTable(TABLE_NAME);
                console.log('🗑️ Dropped existing table');
            } catch (error) {
                // Table might not exist, that's OK
            }
            
            // Recreate the knowledge base
            await this.createKnowledgeBase();
            console.log('✅ Knowledge base rebuilt successfully');
            
        } catch (error) {
            console.error('❌ Error rebuilding knowledge base:', error.message);
            throw error;
        }
    }

    getStatus() {
        return `📊 System Status:
- LanceDB: ${this.db ? '✅ Connected' : '❌ Not connected'}
- Table: ${this.table ? '✅ Available' : '❌ Not available'}
- Documents: ${this.documentCount} chunks
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
                    console.log('\n👋 Goodbye! Thanks for using the LanceDB RAG Chatbot.');
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
        const chatbot = new LanceDBRAGChatbot();
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
module.exports = { LanceDBRAGChatbot };

// Start the application
if (require.main === module) {
    main();
}