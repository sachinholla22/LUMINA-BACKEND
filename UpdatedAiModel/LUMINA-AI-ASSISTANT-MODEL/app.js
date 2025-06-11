import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import readline from 'readline';
import fs from 'fs/promises';
import dotenv from 'dotenv';

dotenv.config();

function shouldLog(){
  return process.env.NODE_ENV !== 'server' && !process.argv.includes('--server');
}

// Global state variables
let conversationHistory = [];
let learningData = {
  supervised: [],
  unsupervised: []
};
let userFeedback = [];
let topicPatterns = new Map();
let responseQuality = new Map();
let learningPhase = 'supervised';
let supervisedThreshold = 50;
let interactionCount = 0;

// Embeddings cache for performance
let embeddingsCache = new Map();
let conversationEmbeddings = [];

const chat = new ChatMistralAI({
  apiKey: process.env.MISTRAL_API_KEY, 
});

// Initialize Mistral embeddings
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed", // Mistral's embedding model
});

async function initialize() {
  await loadLearningData();
  await loadConversationHistory();
  await loadEmbeddingsCache();
  if(shouldLog()){
    console.log(`🧠 Learning Phase: ${learningPhase.toUpperCase()}`);
    console.log(`📊 Interactions: ${interactionCount}`);
    console.log(`🔍 Cached Embeddings: ${embeddingsCache.size}`);
  }
}

async function loadLearningData() {
  try {
    const data = await fs.readFile('learning_data.json', 'utf8');
    const parsed = JSON.parse(data);
    learningData = parsed.learningData || { supervised: [], unsupervised: [] };
    userFeedback = parsed.userFeedback || [];
    topicPatterns = new Map(parsed.topicPatterns || []);
    responseQuality = new Map(parsed.responseQuality || []);
    interactionCount = parsed.interactionCount || 0;
    learningPhase = interactionCount >= supervisedThreshold ? 'unsupervised' : 'supervised';
  } catch (error) {
    if(shouldLog()){
      console.log("📝 Starting with fresh learning data...");
    }
  }
}

async function loadConversationHistory() {
  try {
    const data = await fs.readFile('conversation_history.json', 'utf8');
    conversationHistory = JSON.parse(data) || [];
  } catch (error) {
    if(shouldLog()){
      console.log("📝 Starting with fresh conversation history...");
    }
  }
}

async function loadEmbeddingsCache() {
  try {
    const data = await fs.readFile('embeddings_cache.json', 'utf8');
    const parsed = JSON.parse(data);
    embeddingsCache = new Map(parsed.embeddingsCache || []);
    conversationEmbeddings = parsed.conversationEmbeddings || [];
  } catch (error) {
    if(shouldLog()){
      console.log("📝 Starting with fresh embeddings cache...");
    }
  }
}

async function saveLearningData() {
  const data = {
    learningData: learningData,
    userFeedback: userFeedback,
    topicPatterns: Array.from(topicPatterns.entries()),
    responseQuality: Array.from(responseQuality.entries()),
    interactionCount: interactionCount,
    learningPhase: learningPhase
  };
  
  await fs.writeFile('learning_data.json', JSON.stringify(data, null, 2));
}

async function saveConversationHistory() {
  await fs.writeFile(
    'conversation_history.json', 
    JSON.stringify(conversationHistory, null, 2)
  );
}

async function saveEmbeddingsCache() {
  const data = {
    embeddingsCache: Array.from(embeddingsCache.entries()),
    conversationEmbeddings: conversationEmbeddings
  };
  await fs.writeFile('embeddings_cache.json', JSON.stringify(data, null, 2));
}

// Generate embeddings with caching
async function generateEmbedding(text) {
  const cacheKey = text.substring(0, 100); // Use first 100 chars as cache key
  
  if (embeddingsCache.has(cacheKey)) {
    return embeddingsCache.get(cacheKey);
  }
  
  try {
    const embedding = await embeddings.embedQuery(text);
    embeddingsCache.set(cacheKey, embedding);
    return embedding;
  } catch (error) {
    if(shouldLog()){
      console.error("❌ Error generating embedding:", error.message);
    }
    return null;
  }
}

// Calculate cosine similarity between two embeddings
function cosineSimilarity(vec1, vec2) {
  if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;
  
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

// Find similar conversations using embeddings
async function findSimilarConversations(query, topK = 5) {
  const queryEmbedding = await generateEmbedding(query);
  if (!queryEmbedding) return [];
  
  const similarities = [];
  
  for (let convEmb of conversationEmbeddings) {
    const similarity = cosineSimilarity(queryEmbedding, convEmb.embedding);
    similarities.push({
      ...convEmb,
      similarity: similarity
    });
  }
  
  return similarities
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

// Enhanced research query validation with embeddings
async function validateResearchQuery(query) {
  const researchKeywords = [
    'research', 'study', 'analysis', 'theory', 'methodology',
    'data', 'experiment', 'hypothesis', 'scientific', 'academic',
    'paper', 'journal', 'publication', 'findings', 'literature',
    'review', 'investigation', 'what', 'why', 'how',
    'explain', 'define', 'analyze', 'compare', 'evaluate',
    'discuss', 'examine', 'explore', 'investigate'
  ];
  
  const queryLower = query.toLowerCase();
  
  // Allow greetings and thanks
  if (/^(hi|hello|hey|greetings|thanks|thank you|bye|goodbye)/.test(queryLower)) {
    return true;
  }
  
  // Basic keyword check
  const hasResearchKeywords = researchKeywords.some(keyword => queryLower.includes(keyword)) || 
                             queryLower.length >= 15;
  
  // Enhanced validation using embeddings in unsupervised phase
  if (learningPhase === 'unsupervised' && conversationEmbeddings.length > 0) {
    const similarConversations = await findSimilarConversations(query, 3);
    const avgSimilarity = similarConversations.reduce((sum, conv) => sum + conv.similarity, 0) / similarConversations.length;
    
    // If we have high similarity with past research conversations, consider it research-related
    if (avgSimilarity > 0.7) {
      return true;
    }
  }
  
  // Use learned patterns in unsupervised phase
  if (learningPhase === 'unsupervised') {
    for (let [pattern, isResearch] of topicPatterns) {
      if (queryLower.includes(pattern) && isResearch) {
        return true;
      }
    }
  }
  
  return hasResearchKeywords;
}

// Extract topics and patterns for unsupervised learning
function extractTopicPatterns(query, isResearchRelated) {
  const words = query.toLowerCase().split(/\s+/).filter(word => word.length > 3);
  const patterns = [];
  
  // Extract n-grams
  for (let i = 0; i < words.length - 1; i++) {
    patterns.push(words[i] + ' ' + words[i + 1]);
  }
  
  // Store patterns with their research relevance
  patterns.forEach(pattern => {
    topicPatterns.set(pattern, isResearchRelated);
  });
}

// Enhanced supervised learning with embeddings
async function supervisedLearning(userInput, assistantResponse, feedback = null) {
  const inputEmbedding = await generateEmbedding(userInput);
  const responseEmbedding = await generateEmbedding(assistantResponse);
  
  const dataPoint = {
    input: userInput,
    response: assistantResponse,
    timestamp: new Date().toISOString(),
    feedback: feedback,
    isResearchRelated: await validateResearchQuery(userInput),
    inputEmbedding: inputEmbedding,
    responseEmbedding: responseEmbedding
  };
  
  learningData.supervised.push(dataPoint);
  extractTopicPatterns(userInput, dataPoint.isResearchRelated);
  
  // Store conversation embedding for similarity search
  if (inputEmbedding) {
    conversationEmbeddings.push({
      input: userInput,
      response: assistantResponse,
      embedding: inputEmbedding,
      timestamp: new Date().toISOString(),
      isResearchRelated: dataPoint.isResearchRelated
    });
  }
  
  if(shouldLog()){
    console.log("📚 [Supervised Learning] Data point with embeddings collected");
  }
}

// Enhanced unsupervised learning with semantic clustering
async function unsupervisedLearning(userInput, assistantResponse) {
  const inputEmbedding = await generateEmbedding(userInput);
  const responseEmbedding = await generateEmbedding(assistantResponse);
  
  // Find semantic similarity using embeddings
  const similarity = await calculateSemanticSimilarity(userInput);
  const cluster = await findOrCreateSemanticCluster(userInput, inputEmbedding);
  
  const dataPoint = {
    input: userInput,
    response: assistantResponse,
    timestamp: new Date().toISOString(),
    cluster: cluster,
    patterns: extractFeatures(userInput),
    inputEmbedding: inputEmbedding,
    responseEmbedding: responseEmbedding,
    semanticSimilarity: similarity
  };
  
  learningData.unsupervised.push(dataPoint);
  updateResponseQuality(userInput, assistantResponse);
  
  // Store conversation embedding
  if (inputEmbedding) {
    conversationEmbeddings.push({
      input: userInput,
      response: assistantResponse,
      embedding: inputEmbedding,
      timestamp: new Date().toISOString(),
      cluster: cluster
    });
  }
  
  if(shouldLog()){
    console.log("🔍 [Unsupervised Learning] Semantic pattern analysis completed");
  }
}

// Enhanced similarity calculation using embeddings
async function calculateSemanticSimilarity(input) {
  const inputEmbedding = await generateEmbedding(input);
  if (!inputEmbedding) return 0;
  
  let maxSimilarity = 0;
  
  for (let convEmb of conversationEmbeddings) {
    const similarity = cosineSimilarity(inputEmbedding, convEmb.embedding);
    maxSimilarity = Math.max(maxSimilarity, similarity);
  }
  
  return maxSimilarity;
}

// Create semantic clusters using embeddings
async function findOrCreateSemanticCluster(input, inputEmbedding) {
  if (!inputEmbedding) return `cluster_${Date.now()}`;
  
  const threshold = 0.75; // High similarity threshold for clustering
  
  for (let convEmb of conversationEmbeddings) {
    const similarity = cosineSimilarity(inputEmbedding, convEmb.embedding);
    if (similarity > threshold && convEmb.cluster) {
      return convEmb.cluster;
    }
  }
  
  return `semantic_cluster_${Date.now()}`;
}

//Fallback similarity calculation

// function calculateSimilarity(input) {
//   const inputWords = new Set(input.toLowerCase().split(/\s+/));
//   let maxSimilarity = 0;
  
//   for (let data of learningData.unsupervised) {
//     const dataWords = new Set(data.input.toLowerCase().split(/\s+/));
//     const intersection = new Set([...inputWords].filter(x => dataWords.has(x)));
//     const union = new Set([...inputWords, ...dataWords]);
//     const similarity = intersection.size / union.size;
//     maxSimilarity = Math.max(maxSimilarity, similarity);
//   }
  
//   return maxSimilarity;
// }

// function findOrCreateCluster(input, similarity) {
//   if (similarity > 0.3) {
//     for (let data of learningData.unsupervised) {
//       if (calculateSimilarity(input) > 0.3) {
//         return data.cluster;
//       }
//     }
//   }
//   return `cluster_${Date.now()}`;
// }

function extractFeatures(input) {
  return {
    length: input.length,
    wordCount: input.split(/\s+/).length,
    hasQuestionWords: /\b(what|why|how|when|where|who)\b/i.test(input),
    hasResearchTerms: /\b(study|research|analysis|theory)\b/i.test(input),
    complexity: input.split(/[.!?]/).length
  };
}

function updateResponseQuality(input, response) {
  const key = input.substring(0, 50);
  const currentQuality = responseQuality.get(key) || { count: 0, avgLength: 0 };
  
  currentQuality.count++;
  currentQuality.avgLength = (currentQuality.avgLength + response.length) / 2;
  
  responseQuality.set(key, currentQuality);
}

// Enhanced system prompt with semantic context
async function generateSystemPrompt(currentQuery) {
  let basePrompt = `You are an intelligent research assistant that learns and adapts. 
Current learning phase: ${learningPhase.toUpperCase()}

Core capabilities:
1. Focus on academic and research topics
2. Provide detailed, well-structured responses
3. Engage naturally while maintaining research focus
4. Learn from interactions and improve over time`;

  if (learningPhase === 'unsupervised') {
    // Add learned patterns to prompt
    const topPatterns = Array.from(topicPatterns.entries())
      .filter(([_, isResearch]) => isResearch)
      .slice(0, 10)
      .map(([pattern]) => pattern);
    
    if (topPatterns.length > 0) {
      basePrompt += `\n\nLearned research patterns: ${topPatterns.join(', ')}`;
    }
    
    // Add similar conversation context
    if (currentQuery && conversationEmbeddings.length > 0) {
      const similarConversations = await findSimilarConversations(currentQuery, 3);
      if (similarConversations.length > 0) {
        const contextExamples = similarConversations
          .filter(conv => conv.similarity > 0.5)
          .map(conv => `Q: ${conv.input.substring(0, 100)}...`)
          .join('\n');
        
        if (contextExamples) {
          basePrompt += `\n\nSimilar past queries:\n${contextExamples}`;
        }
      }
    }
  }

  return basePrompt;
}

async function processUserInput(userInput) {
  try {
    interactionCount++;
    
    // Check if we should switch to unsupervised learning
    if (learningPhase === 'supervised' && interactionCount >= supervisedThreshold) {
      learningPhase = 'unsupervised';
      if(shouldLog()){
        console.log("\n🎓 Switching to UNSUPERVISED learning mode with semantic analysis!\n");
      }
    }

    // Validate research query
    const isResearchRelated = await validateResearchQuery(userInput);
    if (!isResearchRelated && !isGreeting(userInput)) {
      const warningMessage = "I specialize in research topics. Could you ask me about an academic subject or research area?";
      if(shouldLog()){
        console.log("\n🤖 Assistant:", warningMessage,"\n");
      }
      return {
        input: userInput,
        response: warning,
        timestamp: new Date().toISOString(),
        isResearchRelated: false
      };
    }

    // Add to conversation history
    conversationHistory.push({ role: 'user', content: userInput });

    // Generate context-aware prompt with semantic context
    const systemPrompt = await generateSystemPrompt(userInput);
    const contextPrompt = conversationHistory
      .slice(-10)
      .map(msg => `${msg.role}: ${msg.content}`)
      .join('\n');

    // Get AI response
    const response = await chat.call([
      new SystemMessage(`${systemPrompt}\n\nRecent context:\n${contextPrompt}`),
      new HumanMessage(userInput)
    ]);

    const assistantResponse = response.text || response.content;

    // Add assistant response to history
    conversationHistory.push({ role: 'assistant', content: assistantResponse });

    let result;

    // Apply appropriate learning method with embeddings
    if (learningPhase === 'supervised') {
      await supervisedLearning(userInput, assistantResponse);
      result = {
        input: userInput,
        response: assistantResponse,
        timestamp: new Date().toISOString(),
        feedback: null,
        isResearchRelated
      };
    } else {
      const inputEmbedding = await generateEmbedding(userInput);
      const cluster = await findOrCreateSemanticCluster(userInput, inputEmbedding);
      const patterns = extractFeatures(userInput);

      await unsupervisedLearning(userInput, assistantResponse);
      result = {
        input: userInput,
        response: assistantResponse,
        timestamp: new Date().toISOString(),
        cluster,
        patterns
      };
    }

    await saveLearningData();
    await saveConversationHistory();
    await saveEmbeddingsCache();

    if(shouldLog()){
      console.log(`\n🤖 Assistant [${learningPhase}]:`, assistantResponse);
      
      // Show semantic similarity info in unsupervised mode
      if (learningPhase === 'unsupervised') {
        const similarity = await calculateSemanticSimilarity(userInput);
        console.log(`🔍 Semantic similarity with past conversations: ${(similarity * 100).toFixed(1)}%`);
      }
      
      // Ask for feedback in supervised phase
      if (learningPhase === 'supervised' && interactionCount % 5 === 0) {
        askForFeedback();
      }
    }
    
    // return assistantResponse;
    return result;

  } catch (error) {
    if(shouldLog()){
      console.error("❌ Error:", error.message);
    }
    throw error;
  }
}

function isGreeting(input) {
  return /^(hi|hello|hey|greetings|thanks|thank you|bye|goodbye)/i.test(input.trim());
}

function askForFeedback() {
  if(shouldLog()){
    console.log("\n💬 Was my response helpful? (Type 'yes', 'no', or just continue with your next question)");
  }
}

async function handleFeedback(feedback) {
  if (['yes', 'no', 'good', 'bad', 'helpful', 'not helpful'].includes(feedback.toLowerCase())) {
    const lastInteraction = learningData.supervised[learningData.supervised.length - 1];
    if (lastInteraction) {
      lastInteraction.userFeedback = feedback;
      userFeedback.push({
        feedback: feedback,
        timestamp: new Date().toISOString(),
        context: lastInteraction.input
      });
      if(shouldLog()){
        console.log("📝 Thank you for your feedback!");
      }
      await saveLearningData();
      return true;
    }
  }
  return false;
}

function displayStats() {
  if(shouldLog()){
    console.log("\n📊 Learning Statistics:");
    console.log(`Interactions: ${interactionCount}`);
    console.log(`Learning Phase: ${learningPhase.toUpperCase()}`);
    console.log(`Supervised Data Points: ${learningData.supervised.length}`);
    console.log(`Unsupervised Data Points: ${learningData.unsupervised.length}`);
    console.log(`Learned Patterns: ${topicPatterns.size}`);
    console.log(`User Feedback Received: ${userFeedback.length}`);
    console.log(`🔍 Cached Embeddings: ${embeddingsCache.size}`);
    console.log(`🧠 Conversation Embeddings: ${conversationEmbeddings.length}\n`);
  }
}

// New function to search similar conversations
async function searchSimilarConversations(query, threshold = 0.5) {
  const similarConversations = await findSimilarConversations(query, 10);
  const filtered = similarConversations.filter(conv => conv.similarity > threshold);
  
  if(shouldLog()){
    console.log(`\n🔍 Found ${filtered.length} similar conversations:`);
    filtered.forEach((conv, index) => {
      console.log(`${index + 1}. [${(conv.similarity * 100).toFixed(1)}%] ${conv.input.substring(0, 80)}...`);
    });
  }
  
  return filtered;
}

// New function to analyze conversation clusters
async function analyzeConversationClusters() {
  const clusters = new Map();
  
  conversationEmbeddings.forEach(conv => {
    if (conv.cluster) {
      if (!clusters.has(conv.cluster)) {
        clusters.set(conv.cluster, []);
      }
      clusters.get(conv.cluster).push(conv);
    }
  });
  
  if(shouldLog()){
    console.log(`\n🎯 Conversation Clusters Analysis:`);
    console.log(`Total clusters: ${clusters.size}`);
    
    Array.from(clusters.entries()).forEach(([clusterId, conversations]) => {
      console.log(`\nCluster ${clusterId}: ${conversations.length} conversations`);
      console.log(`Sample: ${conversations[0].input.substring(0, 100)}...`);
    });
  }
  
  return clusters;
}

export {
  initialize, 
  processUserInput, 
  displayStats, 
  handleFeedback,
  searchSimilarConversations,
  analyzeConversationClusters,
  generateEmbedding,
  findSimilarConversations
};