/*
 * ==============================================================================
 * CODEVERITUS SYSTEM CORE MODULE
 * ==============================================================================
 * File: back-end/models/admins.js
 * Author: ABHIMANYU993
 * Email: abhimanyubadiger1001@gmail.com
 * Project: Codeveritus - AI vs Human Code Classifier
 * Description: Modular component containing system configuration or script details.
 * ==============================================================================
 */

const mongoose = require("mongoose");
const { adminsDB } = require("../dbConnections/db");

// User Schema
const adminsSchema = new mongoose.Schema({
  role: { type: String, required: false, default: "admin" }, // 'user' or 'admin'
  adminname: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

module.exports = adminsDB.model("admins", adminsSchema);