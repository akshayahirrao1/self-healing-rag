import { useState, useRef, useEffect } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { Send, Bot, User, Loader2, Search, BrainCircuit, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: 'user', content: input };
    const aiMsgId = Date.now().toString();
    
    // Store the full AI message state including reasoning steps
    const initialAiMsg = {
      id: aiMsgId,
      role: 'ai',
      content: '',
      reasoning: [],
      isComplete: false
    };

    setMessages(prev => [...prev, userMsg, initialAiMsg]);
    setInput("");
    setIsLoading(true);

    try {
      await fetchEventSource('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.content }),
        onmessage(ev) {
          if (!ev.data) return;
          const parsed = JSON.parse(ev.data);
          
          setMessages(prev => prev.map(msg => {
            if (msg.id === aiMsgId) {
              const newReasoning = [...msg.reasoning];
              
              // Map LangGraph node names to friendly reasoning steps
              let stepLabel = '';
              let status = 'active';
              
              if (parsed.node === 'retrieve') {
                stepLabel = `Retrieving documents for: "${parsed.state.current_query}"`;
              } else if (parsed.node === 'generate') {
                stepLabel = 'Drafting an answer based on context...';
              } else if (parsed.node === 'critic') {
                const decision = parsed.state.decision;
                if (decision === 'accept') {
                  stepLabel = 'Critic accepted the answer!';
                  status = 'accept';
                } else if (decision === 'reject') {
                  stepLabel = `Critic REJECTED draft: ${parsed.state.critique}`;
                  status = 'reject';
                }
              } else if (parsed.node === 'reformulate') {
                stepLabel = 'Reformulating search query to try again...';
              } else if (parsed.node === 'fallback') {
                stepLabel = 'Max retries reached. Triggering fallback response.';
                status = 'reject';
              }
              
              if (stepLabel) {
                newReasoning.push({ label: stepLabel, status, id: Date.now() + Math.random() });
              }

              return {
                ...msg,
                content: parsed.state.answer || '',
                reasoning: newReasoning
              };
            }
            return msg;
          }));
        },
        onclose() {
          setMessages(prev => prev.map(msg => 
            msg.id === aiMsgId ? { ...msg, isComplete: true } : msg
          ));
          setIsLoading(false);
        },
        onerror(err) {
          console.error("EventSource failed:", err);
          setIsLoading(false);
          throw err;
        }
      });
    } catch (err) {
      console.error(err);
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Self-Healing RAG</h1>
        <p>LangGraph Reasoning Visualizer</p>
      </header>

      <div className="chat-container">
        {messages.length === 0 && (
          <div className="text-center text-secondary" style={{ marginTop: '2rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
            Ask a question based on the company policy (e.g., vacation days, remote work).
          </div>
        )}
        
        {messages.map((msg, i) => (
          <div key={msg.id || i} className={`message-wrapper ${msg.role}`}>
            <div className={`bubble ${msg.role}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', fontWeight: 600, color: msg.role === 'user' ? '#fff' : '#a78bfa' }}>
                {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                {msg.role === 'user' ? 'You' : 'AI Agent'}
              </div>
              
              <div>{msg.content}</div>

              {msg.role === 'ai' && msg.reasoning && msg.reasoning.length > 0 && (
                <div className="reasoning-container">
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                    Agent Reasoning Process
                  </div>
                  {msg.reasoning.map(step => {
                    let Icon = BrainCircuit;
                    if (step.label.includes('Retrieving')) Icon = Search;
                    if (step.label.includes('Drafting')) Icon = Loader2;
                    if (step.status === 'accept') Icon = CheckCircle;
                    if (step.status === 'reject') Icon = AlertTriangle;
                    if (step.label.includes('Reformulating')) Icon = RefreshCw;
                    
                    return (
                      <div key={step.id} className={`reasoning-step ${!msg.isComplete && step === msg.reasoning[msg.reasoning.length-1] ? 'active' : step.status}`}>
                        <Icon size={14} className={(!msg.isComplete && step === msg.reasoning[msg.reasoning.length-1] && Icon === Loader2) ? 'spin' : ''} />
                        <span>{step.label}</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-container" onSubmit={handleSubmit}>
        <input 
          type="text" 
          placeholder="Ask a question..." 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="send-btn" disabled={isLoading || !input.trim()}>
          {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
        </button>
      </form>
    </div>
  );
}

export default App;
