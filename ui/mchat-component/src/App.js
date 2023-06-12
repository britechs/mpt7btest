import React from 'react';
import Tabs from "./components/Tabs";
import Tab from "./components/Tab";
import "./App.css";
import QuestionComponent from './components/Question'
import IngestDocument from './components/IngestDocument'
import DocumentList from './components/DocumentList'

function App() {
  return (
    <div>
      <h1>Mortgage Chatbot</h1>

      <Tabs>
        <div label="Ask Question">
          <QuestionComponent></QuestionComponent>
        </div>
        <div label="Ingest Document">
          <IngestDocument></IngestDocument>
        </div>
        <div label="Available Documents">
          <DocumentList></DocumentList>
        </div>
      </Tabs>
    </div>
  );
}

export default App;