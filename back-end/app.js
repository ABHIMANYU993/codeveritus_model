// // app.js
//
// require("dotenv").config();
// const express = require("express");
// const cors = require("cors");
// const { connectDB } = require("./dbConnections/db");
// const usersRoute = require("./routes/usersRoutes");
// const adminsRoute = require("./routes/adminsRoutes");
//
// const app = express();
// app.use(express.json());
// app.set("trust proxy", true);
//
// // CORS configuration (your existing code is fine)
// app.use(
//   cors({
//     origin: (origin, cb) =>
//       !origin || origin.startsWith("https://codeveritus.makeatron.in")
//         ? cb(null, true)
//         : cb(new Error("CORS blocked")),
//     credentials: true,
//   })
// );
//
// // Test endpoint
// app.get("/", (req, res) => res.send("✅ CODEVERITUS BACKEND (Docker container running)"));
//
// // --- CORRECTED API ROUTES ---
// app.use("/api/users", usersRoute);
//
// // Use only ONE entry point for all admin routes.
// app.use("/api/admins", adminsRoute);
// // REMOVED: app.use("/api/admins/fetch", jwtAuthenticator, adminsRoute);
//
// // Connect to DB
// connectDB();
//
// const PORT = process.env.PORT || 4000;
// app.listen(PORT, "0.0.0.0", () => {
//   console.log(`🚀 Backend server running on port ${PORT}`);
// });



// Polyfill for SlowBuffer to support older modules (like buffer-equal-constant-time) in Node.js 22+
const buffer = require('buffer');
if (!buffer.SlowBuffer) {
  buffer.SlowBuffer = buffer.Buffer;
}

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const { connectDB, closeDB } = require("./dbConnections/db");
const usersRoute = require("./routes/usersRoutes");
const adminsRoute = require("./routes/adminsRoutes");
const jwtAuthenticator = require("./middleware/jwtAuthenticator");

const app = express();
app.use(express.json());
app.set("trust proxy", true);

// CORS: restrict to your frontend domain
app.use(
  cors({