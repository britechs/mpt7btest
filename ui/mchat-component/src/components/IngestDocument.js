import React, { useState, useEffect } from 'react';
import AppConfig from './AppConfig';

const IngestDocument = () => {
  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [responseMessage, setResponseMessage] = useState('');
  const backendurl = AppConfig.backendURL + '/api/ingest';

  useEffect(() => {
    const storedData = JSON.parse(localStorage.getItem('ingestDocumentData'));

    if (storedData) {
      setUrl(storedData.url);
      setTitle(storedData.title);
      setContent(storedData.content);
    }
  }, []);

  useEffect(() => {
    console.debug("update to local storage");
    const dataToStore = JSON.stringify({ url, title, content });
    localStorage.setItem('ingestDocumentData', dataToStore);
  }, [url, title, content, localStorage]);

  const handleSubmit = async () => {
    try {
      const response = await fetch(backendurl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning':true
        },
        body: JSON.stringify({ url, title, content }),
      });

      if (response.status === 200) {
        setResponseMessage('Success');
      } else {
        setResponseMessage('Error');
      }
    } catch (error) {
      console.error('Error:', error);
      setResponseMessage('Error');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <div>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="URL"
          style={{ width: '80%', margin: '10px' }}
        />
      </div>
      <div>
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          style={{ width: '80%', margin: '10px' }}
        />
      </div>
      <div>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={20}
          placeholder="Content"
          style={{ width: '80%', margin: '10px' }}
        />
      </div>
      <div>
        <button onClick={handleSubmit} style={{ margin: '10px' }}>Submit</button>
      </div>
      <div>{responseMessage}</div>
    </div>
  );
};

export default IngestDocument;
