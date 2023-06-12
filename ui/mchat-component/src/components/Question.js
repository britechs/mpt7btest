import React, { useState } from 'react';
import AppConfig from './AppConfig'

const QuestionComponent = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const backendurl = AppConfig.backendURL + '/api/qa';

  const handleSubmit = async () => {
    try {
      const response = await fetch(backendurl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning':true
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setAnswer(data.message.answer);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <div>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          placeholder="Question"
          style={{ haligh:'center', width: '80%', margin: '10px' }}
        />
      </div>
      <div>
        <button onClick={handleSubmit} style={{ margin: '10px' }}>Submit</button>
      </div>
      <div>
        <textarea
          value={answer}
          rows={10}
          placeholder="Answer"
          readOnly
          style={{ haligh:'center', width: '80%', margin: '10px' }}
        />
      </div>
    </div>
  );
};

export default QuestionComponent;
