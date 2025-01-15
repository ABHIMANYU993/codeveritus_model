import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { FaArrowLeft, FaChevronDown, FaChevronUp } from "react-icons/fa";
import "./Results.css";

const API_URL = process.env.REACT_APP_API_URL;

const ResultsPage = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const navigate = useNavigate();

  // Format as "xx.xx% Of code similar to AI-generated code." or "...Human..."
  const parsePrediction = (rawPrediction) => {
    let ai = null, human = null;
    if (typeof rawPrediction === "string") {
      const aiMatch = rawPrediction.match(/AI:\s*([\d.]+%)/i);
      const humanMatch = rawPrediction.match(/Human:\s*([\d.]+%)/i);
      ai = aiMatch ? aiMatch[1] : null;
      human = humanMatch ? humanMatch[1] : null;
    } else if (typeof rawPrediction === "object" && rawPrediction !== null) {
      ai = rawPrediction.AI || null;
      human = rawPrediction.Human || null;
    }
    // Pick highest value and show that one
    if (ai && (!human || parseFloat(ai) >= parseFloat(human))) {
      return `${ai} Of code similar to AI-generated code.`;
    }
    if (human) {
      return `${human} Of code similar to Human-generated code.`;
    }
    return "N/A";
  };

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const token = localStorage.getItem("jwtToken");
        if (!token) {
          navigate("/login");
          return;
        }
        const response = await axios.get(
          `${API_URL}/api/admins/fetch/codeSubmissions`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        const formattedResults = response.data.map((item) => {
          const rawPrediction = item.detectionResult?.predictions?.[0];
          return {
            ...item,
            codes: item.codes || [],
            prediction: parsePrediction(rawPrediction),
          };
        });