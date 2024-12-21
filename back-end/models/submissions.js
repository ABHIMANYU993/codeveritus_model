const mongoose = require("mongoose");
const { submissionsDB } = require("../dbConnections/db");

const UserCodeSchema = new mongoose.Schema({
  username: { type: String, required: true },
  codes: {
    type: [String],
    default: [],
  },
  detectionResult: {
    type: mongoose.Schema.Types.Mixed,
    default: {},
  },
  submittedAt: {
    type: Date,
    default: Date.now, // Use Date type, auto-generated when doc is created
  },
});

module.exports = submissionsDB.model("submissions", UserCodeSchema);