/*
 * ==============================================================================
 * CODEVERITUS SYSTEM CORE MODULE
 * ==============================================================================
 * File: back-end/middleware/corsConfig.js
 * Author: ABHIMANYU993
 * Email: abhimanyubadiger1001@gmail.com
 * Project: Codeveritus - AI vs Human Code Classifier
 * Description: Modular component containing system configuration or script details.
 * ==============================================================================
 */

// middleware/corsConfig.js
const cors = require("cors");

const corsMiddleware = cors({
  //origin: 'http://localhost:3000',
  origin: [
    'http://192.168.31.77:3000',
    'https://192.168.31.77:3000',
    'https://codeveritus.makeatron.in',
    'https://apicodeveritus.makeatron.in'
  ],
  methods: ["GET", "POST", "PUT", "DELETE"],
  credentials: true,
});

module.exports = corsMiddleware;