import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFacebook, faTwitter, faLinkedin, faGithub } from '@fortawesome/free-brands-svg-icons'; 

const Footer = () => {
  return (
    <footer className="text-center py-4 bg-dark text-white">
      <p className="quote mb-3" style={{ fontStyle: 'italic', fontSize: '1.2rem' }}>
        "CodeVeritus—because originality fuels innovation."
      </p>
      <div className="social-icons mb-3">
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="mx-2">
          <FontAwesomeIcon icon={faFacebook} size="2x" style={{ color: '#3b5998' }} />
        </a>
        <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="mx-2">
          <FontAwesomeIcon icon={faTwitter} size="2x" style={{ color: '#1da1f2' }} />