import React, { useState, useRef, useEffect } from 'react';
import { chatAPI } from '../services/api';

const ChatWindow = ({ messages, onNewMessage, currentChat, isNewChat, onClearAll, isLoadingHistory }) => {
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [tempUserMessage, setTempUserMessage] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Clear temp messages when chat changes or history loads
  useEffect(() => {
    if (!isLoadingHistory) {
      setTempUserMessage(null);
    }
  }, [currentChat, isLoadingHistory]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, tempUserMessage]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || loading) return;

    const timestamp = new Date().toISOString();
    const messageContent = inputMessage.trim();

    const userMessage = {
      id: `temp-${Date.now()}-user`, // Mark as temporary
      role: 'user',
      content: messageContent,
      timestamp,
      isTemp: true, // Add temp flag
    };

    // Store temp message and clear input immediately
    setTempUserMessage(userMessage);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await chatAPI.sendMessage(messageContent, currentChat?._id);

      // Clear temporary message immediately after successful response
      setTempUserMessage(null);

      // Assistant message
      const assistantMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date().toISOString(),
      };

      // Send both messages to parent - user message WITHOUT temp flag
      const persistentUserMessage = {
        ...userMessage,
        id: `${Date.now()}-user`, // Remove temp prefix
        isTemp: undefined, // Remove temp flag
      };

      onNewMessage(
        {
          chat_id: response.data.chat_id,
          messages: [persistentUserMessage, assistantMessage],
        },
        response.data.chat_id
      );

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Clear temporary message on error
      setTempUserMessage(null);
      
      const errorMessage = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        content: '⚠️ Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      };
      
      // On error, still save the user message but with error response
      const persistentUserMessage = {
        ...userMessage,
        id: `${Date.now()}-user`,
        isTemp: undefined,
      };

      onNewMessage(
        {
          chat_id: currentChat?._id || `${Date.now()}`,
          messages: [persistentUserMessage, errorMessage],
        }
      );
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const displayMessages = React.useMemo(() => {
    const cleanMessages = messages.filter(msg => 
      !msg.isTemp && 
      !msg.id?.startsWith('temp-') &&
      msg.content && 
      !msg.content.includes('(sending...)')
    );
    
    return tempUserMessage ? [...cleanMessages, tempUserMessage] : cleanMessages;
  }, [messages, tempUserMessage]);

  const suggestedQuestions = [
    'What is the leave policy?',
    'How do I apply for sick leave?',
    'What are the working hours?',
    'Tell me about employee benefits',
  ];

  const handleSuggestionClick = (question) => {
    setInputMessage(question);
    inputRef.current?.focus();
  };

  // Show loading state while history is being fetched
  if (isLoadingHistory && messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col h-full bg-white">
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-500">Loading chat history...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <h1 className="text-lg font-semibold text-gray-900">
          {isNewChat ? 'Start New Conversation' : 'Chat History'}
        </h1>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {displayMessages.length === 0 && isNewChat ? (
          <div className="flex flex-col items-center justify-center text-center text-gray-500 h-full">
            <h3 className="text-xl font-semibold mb-3">Welcome to HR Assistant 👋</h3>
            <p className="mb-6">Ask me about company policies, leaves, or benefits.</p>
            <div className="space-y-3">
              {suggestedQuestions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSuggestionClick(q)}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50 transition"
                  disabled={loading || isLoadingHistory}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {displayMessages.map((msg, index) => {
              const role = msg.role || (msg.isUser ? 'user' : 'assistant');
              const isTemp = msg.isTemp || msg.id === tempUserMessage?.id;
              
              return (
                <div
                  key={msg.id || `${role}-${index}-${msg.timestamp}`}
                  className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'} ${
                    isTemp ? 'opacity-70' : ''
                  }`}
                >
                  <div
                    className={`flex items-start space-x-3 max-w-3xl ${
                      role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                    }`}
                  >
                    <div
                      className={`w-8 h-8 flex items-center justify-center rounded-lg ${
                        role === 'user' ? 'bg-blue-600' : 'bg-purple-600'
                      } ${isTemp ? 'animate-pulse' : ''}`}
                    >
                      <svg
                        className="w-4 h-4 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        {role === 'user' ? (
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                          />
                        ) : (
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                          />
                        )}
                      </svg>
                    </div>

                    <div
                      className={`rounded-lg px-4 py-3 max-w-lg ${
                        role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-800'
                      } ${isTemp ? 'animate-pulse' : ''}`}
                    >
                      <div className="whitespace-pre-wrap text-sm">
                        {msg.content}
                        {isTemp && (
                          <span className="ml-2 text-xs opacity-70">(sending...)</span>
                        )}
                      </div>
                      {msg.source && !isTemp && (
                        <div className="mt-2 pt-2 border-t border-gray-200 border-opacity-20">
                          <p className="text-xs opacity-70">
                            Source: {msg.source}
                            {msg.confidence && ` • Confidence: ${(msg.confidence * 100).toFixed(1)}%`}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
            {loading && !tempUserMessage && (
              <div className="flex justify-start">
                <div className="flex items-start space-x-3 max-w-3xl">
                  <div className="w-8 h-8 flex items-center justify-center rounded-lg bg-purple-600">
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    </svg>
                  </div>
                  <div className="bg-gray-100 rounded-lg px-4 py-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: '0.2s' }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: '0.4s' }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Box */}
      <div className="border-t border-gray-200 px-6 py-4 bg-white">
        <form onSubmit={sendMessage} className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage(e);
                }
              }}
              placeholder="Type your message here... (Shift+Enter for new line)"
              className="w-full border border-gray-300 rounded-lg px-4 py-3 resize-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm pr-12"
              rows="1"
              disabled={loading || isLoadingHistory}
            />
            {inputMessage && (
              <button
                type="button"
                onClick={() => setInputMessage('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={loading || !inputMessage.trim() || isLoadingHistory}
            className="bg-blue-600 hover:bg-blue-700 disabled:hover:bg-blue-600 text-white p-3 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </form>
        
        {/* Quick Suggestions */}
        {displayMessages.length > 0 && !loading && (
          <div className="mt-3 flex flex-wrap gap-2">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(question)}
                className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition border border-gray-200"
                disabled={loading || isLoadingHistory}
              >
                {question}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatWindow;