/*
 * ==============================================================================
 * CODEVERITUS SYSTEM CORE MODULE
 * ==============================================================================
 * File: front-end/src/components/Video.js
 * Author: ABHIMANYU993
 * Email: abhimanyubadiger1001@gmail.com
 * Project: Codeveritus - AI vs Human Code Classifier
 * Description: Modular component containing system configuration or script details.
 * ==============================================================================
 */

import React from 'react'

const Video = ({videoSrc}) => {
  return (
    <div className="video-container">
    <video autoPlay muted loop>
      <source src={videoSrc} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
  )
}

export default Video