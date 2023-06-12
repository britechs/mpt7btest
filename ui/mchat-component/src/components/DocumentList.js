import React, { useState } from 'react';
import AppConfig from './AppConfig'

const DocumentList = () => {
  const [documents, setDocuments] = useState('');
  const backendurl = AppConfig.backendURL + '/api/docs';
  const backendurl2 = AppConfig.backendURL + '/api/resetdocs';

  const handleRefresh = async () => {
    try {
      const response = await fetch(backendurl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning':true
        },
      });

      const data = await response.json();
      const documentNames = data.message.join('\n');
      setDocuments(documentNames);
    } catch (error) {
      console.error('Error:', error);
      setDocuments('');
    }
  };
  const handleReset = async () => {
    try {
      const response = await fetch(backendurl2, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning':true
        },
      });

      if (response.status === 200) {
        setDocuments("Reseted");
      } else {
        alert("Failed to reset")
      }
      
    } catch (error) {
      console.error('Error:', error);
    }
  };  
  
  return (
    <div style={{ padding: '20px' }}>
      <div>
        <textarea
          value={documents}
          rows={10}
          placeholder="Document List"
          readOnly
          style={{ width: '80%', margin: '10px' }}
        />
      </div>
      <div>
        <button onClick={handleRefresh} style={{margin: '10px' }}>Refresh</button>
        <button onClick={handleReset} style={{margin: '10px' }}>Reset</button>
      </div>
    </div>
  );
};

export default DocumentList;
