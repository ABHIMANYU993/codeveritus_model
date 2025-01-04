import React from 'react';
import CountUp from 'react-countup';
import { useInView } from 'react-intersection-observer';

const StatsSection = () => {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section
      className="stats-section text-white py-5"
      style={{
        background: 'linear-gradient(135deg, #2c3e52, #fd746a)',
        color: 'white',
      }}
    >
      <h2 className="fw-bold text-center mb-3">The Code Authenticity Crisis is Real—and Accelerating</h2>
      <p className="lead text-center">2022-2024 Insights You Can’t Ignore</p>
      <div className="row mt-5" ref={ref}>
        <div className="col-md-4 text-center">
          <h3>
            {inView ? (
              <CountUp
                end={5000000}
                duration={2.5}
                separator=","