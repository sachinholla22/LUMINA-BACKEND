import readline from 'readline';
import { 
  initialize, 
  processUserInput, 
  displayStats, 
  handleFeedback,
  searchSimilarConversations,
  analyzeConversationClusters 
} from '../app.js';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function startConversation() {
  try {
    console.log("🚀 Initializing AI Research Assistant...");
    await initialize();
    
    console.log("\n🔬 Welcome to your AI Research Assistant with Machine Learning!");
    console.log("📚 I learn from our conversations to provide better assistance.");
    console.log("\n📋 Available commands:");
    console.log("  • 'stats' - View learning progress and statistics");
    console.log("  • 'search [query]' - Find similar past conversations");
    console.log("  • 'clusters' - Analyze conversation clusters");
    console.log("  • 'help' - Show this help message");
    console.log("  • 'exit' - End the conversation");
    console.log("\n💡 Just ask me any research-related question to get started!\n");
    
  } catch (error) {
    console.error("❌ Failed to initialize:", error.message);
    process.exit(1);
  }
  
  const askQuestion = () => {
    rl.question('You: ', async (input) => {
      const trimmedInput = input.trim();
      
      if (!trimmedInput) {
        askQuestion();
        return;
      }
      
      try {
        // Handle exit command
        if (trimmedInput.toLowerCase() === 'exit') {
          console.log('\n🎓 Thank you for helping me learn! Goodbye!\n');
          displayStats();
          rl.close();
          return;
        }
        
        // Handle stats command
        if (trimmedInput.toLowerCase() === 'stats') {
          displayStats();
          askQuestion();
          return;
        }
        
        // Handle help command
        if (trimmedInput.toLowerCase() === 'help') {
          showHelp();
          askQuestion();
          return;
        }
        
        // Handle search command
        if (trimmedInput.toLowerCase().startsWith('search ')) {
          const query = trimmedInput.substring(7).trim();
          if (query) {
            await searchSimilarConversations(query);
          } else {
            console.log("📝 Usage: search [your query]");
          }
          askQuestion();
          return;
        }
        
        // Handle clusters command
        if (trimmedInput.toLowerCase() === 'clusters') {
          await analyzeConversationClusters();
          askQuestion();
          return;
        }

        // Check if input is feedback first
        const isFeedback = await handleFeedback(trimmedInput);
        if (!isFeedback) {
          // Process as regular user input
          await processUserInput(trimmedInput);
        }
        
      } catch (error) {
        console.error("❌ Error processing input:", error.message);
        console.log("💡 Please try again or type 'help' for available commands.");
      }
      
      askQuestion();
    });
  };

  askQuestion();
}

function showHelp() {
  console.log("\n📖 AI Research Assistant Help:");
  console.log("──────────────────────────────────");
  console.log("🔍 Research Questions: Ask me about any academic topic, theory, or research area");
  console.log("📊 stats - View learning statistics and progress");
  console.log("🔎 search [query] - Find similar conversations from history");
  console.log("🎯 clusters - Analyze conversation topic clusters"); 
  console.log("❓ help - Show this help message");
  console.log("🚪 exit - End the conversation and show final stats");
  console.log("\n💡 Examples:");
  console.log("  • What is machine learning?");
  console.log("  • Explain quantum computing principles");
  console.log("  • search machine learning");
  console.log("  • clusters");
  console.log("──────────────────────────────────\n");
}

// Enhanced error handling and graceful shutdown
function handleShutdown() {
  console.log('\n\n👋 Shutting down gracefully...');
  displayStats();
  rl.close();
  process.exit(0);
}

// Start the application
startConversation().catch(error => {
  console.error("💥 Application startup error:", error);
  console.error("Stack trace:", error.stack);
  process.exit(1);
});

// Handle various shutdown signals
process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);
process.on('SIGQUIT', handleShutdown);

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('💥 Uncaught Exception:', error);
  handleShutdown();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
  handleShutdown();
});