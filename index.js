const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { glob } = require('glob');
const OpenAI = require('openai');
require('dotenv').config();

// Using LangChain.js for document processing only
const { DirectoryLoader } = require('langchain/document_loaders/fs/directory');
const { TextLoader } = require('langchain/document_loaders/fs/text');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');

// Configuration
const MODEL = "gpt-4o-mini";
const DB_NAME = "vector_db";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

class RAGChatbot {
    constructor() {
        this.vectorstore = null;
        this.openai = null;
        this.chatHistory = [];
    }

    async initialize() {
        // Initialize OpenAI client
        this.openai = new OpenAI({
            apiKey: OPENAI_API_KEY,
        });

        try {
            console.log('Initializing RAG Chatbot...');
            console.log('='.repeat(50));
            
            const documents = await this.loadDocuments();
            await this.createVectorStore(documents);
            
            console.log('='.repeat(50));
            console.log('RAG Chatbot initialized successfully!');
            console.log('You can now ask questions about Insurellm documents.');
            console.log('Type "exit" or "quit" to end the conversation.');
            console.log('='.repeat(50));
            
        } catch (error) {
            console.error('Failed to initialize chatbot:', error);
            throw error;
        }
    }

    async loadDocuments() {
        console.log('Loading documents from knowledge-base...');
        
        try {
            const folders = await glob('knowledge-base/*', { onlyDirectories: true });
            const documents = [];

            for (const folder of folders) {
                const docType = path.basename(folder);
                console.log(`Processing folder: ${docType}`);
                
                const loader = new DirectoryLoader(folder, {
                    '.md': (path) => new TextLoader(path),
                });
                
                const folderDocs = await loader.load();
                
                // Add document type metadata
                folderDocs.forEach(doc => {
                    doc.metadata.doc_type = docType;
                    documents.push(doc);
                });
            }

            console.log(`Loaded ${documents.length} documents`);
            return documents;
        } catch (error) {
            console.error('Error loading documents:', error);
            throw error;
        }
    }

    async createVectorStore(documents) {
        console.log('Creating text chunks...');
        
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const chunks = await textSplitter.splitDocuments(documents);
        console.log(`Created ${chunks.length} text chunks`);

        // Display document types found
        const docTypes = [...new Set(chunks.map(chunk => chunk.metadata.doc_type))];
        console.log(`Document types found: ${docTypes.join(', ')}`);

        console.log('Creating embeddings and vector store...');
        
        const embeddings = new OpenAIEmbeddings({
            openAIApiKey: OPENAI_API_KEY,
        });

        this.vectorstore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

        console.log('Vector store created successfully');
        return this.vectorstore;
    }

    async retrieveRelevantDocuments(query, k = 4) {
        const retriever = this.vectorstore.asRetriever({ k });
        const relevantDocs = await retriever.getRelevantDocuments(query);
        return relevantDocs;
    }

    formatChatHistory() {
        return this.chatHistory.map(msg => {
            return `${msg.role === 'user' ? 'Human' : 'Assistant'}: ${msg.content}`;
        }).join('\n');
    }

    async chat(message) {
        try {
            // Retrieve relevant documents
            const relevantDocs = await this.retrieveRelevantDocuments(message);
            
            // Format context from retrieved documents
            const context = relevantDocs.map((doc, index) => {
                return `Document ${index + 1} (${doc.metadata.doc_type}):\n${doc.pageContent}`;
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

            // Keep chat history manageable (last 10 exchanges)
            if (this.chatHistory.length > 20) {
                this.chatHistory = this.chatHistory.slice(-20);
            }

            return fullResponse;

        } catch (error) {
            console.error('Error during chat:', error);
            return 'Sorry, I encountered an error processing your question. Please try again.';
        }
    }

    async startTerminalInterface() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        const askQuestion = () => {
            rl.question('\n🤖 You: ', async (input) => {
                const message = input.trim();
                
                if (message.toLowerCase() === 'exit' || message.toLowerCase() === 'quit') {
                    console.log('\n👋 Goodbye! Thanks for using the RAG Chatbot.');
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
        const chatbot = new RAGChatbot();
        await chatbot.initialize();
        await chatbot.startTerminalInterface();
    } catch (error) {
        console.error('Fatal error:', error);
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

// Start the application
if (require.main === module) {
    main();
}