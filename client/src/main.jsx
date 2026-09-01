import React from 'react'
import ReactDOM from 'react-dom/client'
import './styles.css'

function App() {
  return (
    <main className="app-shell">
      <h1>Group 27 Project</h1>
      <p>Project starter structure is ready.</p>
      <ul>
        <li>Frontend: React + Vite</li>
        <li>Backend: Express</li>
        <li>Workspace: ready for development</li>
      </ul>
    </main>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
