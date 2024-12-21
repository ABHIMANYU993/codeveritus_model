const express = require("express");
const router = express.Router();
const adminsController = require("../controllers/adminsControllers");

// Routes connected to controller functions
router.post("/signup", adminsController.adminsSignup);
router.post("/login", adminsController.adminsLogin);
router.get("/codeSubmissions", adminsController.seeCode);
module.exports = router;




// const express = require("express");
// const router = express.Router();
// const adminsController = require("../controllers/adminsControllers");
// const jwtAuthenticator = require("../middleware/jwtAuthenticator"); // 1. Import the middleware
//
// // Public routes (no token needed)
// router.post("/signup", adminsController.adminsSignup);
// router.post("/login", adminsController.adminsLogin);
//
// // Protected route (token required)
// // 2. Add middleware and correct the path to match the frontend API call
// router.get("/fetch/codeSubmissions", jwtAuthenticator, adminsController.seeCode);
//
// module.exports = router;