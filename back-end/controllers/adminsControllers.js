//REQUIREMENTS FOR CURRENT MODULE
require("dotenv").config();
const Admin = require("../models/admins");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const UserCode = require("../models/submissions");
exports.adminsSignup = async (req, res) => {
  const { role, adminname, email, password } = req.body;

  try {
    // Check if username or email exists
    const existingUser = await Admin.findOne({ email });
    if (existingUser)
      return res.status(400).json({ message: "User already exists" });

    // Hash password and save new user
    const hashedPassword = await bcrypt.hash(
      password,
      parseInt(process.env.SALTROUNDS)
    );
    const newUser = new Admin({
      role,
      adminname,
      email,
      password: hashedPassword,