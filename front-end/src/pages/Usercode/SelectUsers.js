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