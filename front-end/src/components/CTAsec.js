import React from 'react';
import { useNavigate } from 'react-router-dom'; 
const CTAsec = () => {
  const navigate = useNavigate(); 
  const handleUploadClick = () => {
    navigate('/login'); 
  };

  return (
    <section
      className="text-center py-5"
      style={{
        background: 'linear-gradient(135deg, #2c3e52, #fd746a)',
        color: 'white',
      }}