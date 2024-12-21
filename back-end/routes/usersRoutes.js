const express = require("express");
const router = express.Router();
const usersController = require("../controllers/usersControllers");

router.post("/signup", usersController.usersSignup);
router.post("/login", usersController.usersLogin);
router.post("/submit", usersController.codeSubmit);
module.exports = router;




// const express = require("express");
// const router = express.Router();
// const usersController = require("../controllers/usersControllers");
// const jwtAuthenticator = require("../middleware/jwtAuthenticator"); // 1. Import the middleware
//
// // Public routes
// router.post("/signup", usersController.usersSignup);
// router.post("/login", usersController.usersLogin);
//
// // Protected route (requires a logged-in user)
// // 2. Add middleware to secure this endpoint
// router.post("/submit", jwtAuthenticator, usersController.codeSubmit);
//
// module.exports = router;