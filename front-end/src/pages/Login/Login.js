import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Video from '../../components/Video';
import './Login.css';




const Login = ({ videoSrc }) => {
  const [activeTab, setActiveTab] = useState('user');
  const [isSignUp, setIsSignUp] = useState(false);
  const [credentials, setCredentials] = useState({ username: '', password: '', email: '' });
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false); // Add this line
  const navigate = useNavigate();
  const API_URL = process.env.REACT_APP_API_URL;

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCredentials({ ...credentials, [name]: value });
  };

  // Handle login submission
  const handleLogin = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');
    setIsLoading(true); // Show spinner



    try {
      const endpoint = activeTab === 'user' ? 'api/users/login' : 'api/admins/login';



      const response = await axios.post(`${API_URL}/${endpoint}`, {
        email: credentials.email,
        password: credentials.password,
      });

      if (response.status === 200) {
        localStorage.setItem('username', response.data.username || response.data.adminname);
        localStorage.setItem('jwtToken', response.data.jwtToken);
        activeTab === 'user' ? navigate('/companyinfo') : navigate('/results');
      }
    } catch (error) {
      setErrorMessage(error.response?.data?.message || 'Invalid email or password');
    } finally {
      setIsLoading(false); // Hide spinner
    }
  };

  // Handle signup submission (unchanged)
  const handleSignUp = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const endpoint = activeTab === 'user' ? 'api/users/signup' : 'api/admins/signup';
      const response = await axios.post(`${API_URL}/${endpoint}`, {
        ...(activeTab === 'admin'
          ? { adminname: credentials.username }
          : { username: credentials.username }),
        email: credentials.email,