require("dotenv").config();
const axios = require("axios");
const User = require("../models/users");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const UserCode = require("../models/submissions");


const MODEL_API = process.env.MODEL_API;
exports.usersSignup = async (req, res) => {

  const { role, username, email, password } = req.body;

  try {
    const existingUser = await User.findOne({ email });
    if (existingUser)
      return res.status(400).json({ message: "User already exists" });

    const hashedPassword = await bcrypt.hash(
      password,
      parseInt(process.env.SALTROUNDS)
    ); // FIX: `parseInt()`
    const newUser = new User({
      role,
      username,
      email,
      password: hashedPassword,
    });

    await newUser.save();
    res.status(201).json({ message: `${username} registered successfully` });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({ message: "Server error" });
  }
};

exports.usersLogin = async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email }); // FIX: `User` variable case
    if (!user) return res.status(400).json({ message: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: "Invalid credentials" });
    }

    const token = jwt.sign(
      { userId: user._id, role: user.role },
      process.env.JWT_SECRET_TOKEN
    );

    res.json({
      message: `${user.role} login successful`,
      username: user.username,
      jwtToken: token,
    });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error" });
  }
};

// API: Submit or update user codes
exports.codeSubmit = async (req, res) => {
  try {