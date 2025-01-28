import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import axios from "axios";
import { ReactTyped as Typed } from "react-typed";
import Video from "../../components/Video";
import { Link } from "react-router-dom";
import { FaArrowLeft } from "react-icons/fa";
import "./SelectUsers.css";

const SelectUsersPage = ({ videoSrc, currentUser }) => { // Add currentUser as prop
  const [codeInput, setCodeInput] = useState("");
  const [submittedCode, setSubmittedCode] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL;
  const handleCodeChange = (event) => {
    setCodeInput(event.target.value);
    setSubmittedCode(null);
  };

  const handleSubmit = async () => {
    const username = localStorage.getItem('username');
    if (!username) {
      setErrorMessage("You must be logged in to submit code.");
      return;
    }

    setIsSubmitting(true); // Start submitting
    setErrorMessage("");

    try {
      const userCode = {
        username: username,
        code_samples: [codeInput],
      };

      await axios.post(`${API_URL}/api/users/submit`, userCode);
      setSubmittedCode(codeInput);
      setCodeInput("");
    } catch (error) {
      console.error("Error submitting code:", error);
      setErrorMessage("Failed to submit the code. Please try again.");
    } finally {
      setIsSubmitting(false); // End submitting
    }
  };

  return (
    <div className="full-page-container">
      <div className="content animate-fade-in">
        <h1 className="title animate-slide-down">Code Entry Portal</h1>
        <Video videoSrc="184815-874271897_medium.mp4" />

        <Typed
          className="tagline animate-typed"
          strings={["Empowering Your Coding Journey!"]}
          typeSpeed={50}
          backSpeed={30}
          loop={false}
        />

        <div className="code-entry">
          <textarea
            className="form-control"
            value={codeInput}
            onChange={handleCodeChange}
            placeholder="Enter your code here..."
            rows="5"
          />
          <button