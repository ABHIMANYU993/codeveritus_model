import React from "react";
import { Link } from "react-router-dom";
import '../styles/Navbar.css'; 
const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
      <div className="container">
        <Link className="navbar-brand d-flex align-items-center" to="/">
          <img 
            src="logot.png" 
            alt="CodeVeritus Logo" 
            style={{ width: "40px", height: "40px", marginRight: "10px" }} 
          />
          CodeVeritus
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
        >