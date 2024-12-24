const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    if (mongoose.connection.readyState === 1) {
      console.log("MongoDB already connected");
      return;
    }

    await mongoose.connect(process.env.MONGO_URI);
    console.log("Database connected Successfully!!");
  } catch (error) {
    console.error(`MongoDB connection error: ${error.message}`);
    process.exit(1);
  }
};


// --- NEW FUNCTION: Gracefully closes the MongoDB connection ---
const closeDB = async () => {
  try {
    // Check if the connection is open before trying to close it
    if (mongoose.connection.readyState === 1) {
      await mongoose.connection.close();
      console.log("✅ MongoDB connection closed successfully.");
    }
  } catch (error) {
    console.error(`Error closing MongoDB connection: ${error.message}`);
  }
};

const adminsDB = mongoose.connection.useDb("adminsDB");
const usersDB = mongoose.connection.useDb("usersDB");
const submissionsDB = mongoose.connection.useDb("submissionsDB");

// Add closeDB to the exports
module.exports = { connectDB, closeDB, adminsDB, usersDB, submissionsDB };