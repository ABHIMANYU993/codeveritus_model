import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './CompanyInfo.css';
const CompanyInfo = () => {
  const navigate = useNavigate(); 

  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");

  const formatTime = (date) => {
    let hours = date.getHours();
    let minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    const minutesStr = minutes < 10 ? '0' + minutes : minutes;
    return `${hours}:${minutesStr} ${ampm}`;
  };

 
  useEffect(() => {
    const now = new Date();