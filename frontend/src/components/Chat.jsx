import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './Sidebar';
import ChatWindow from './ChatWindow';
import { chatAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';

const Chat = () => {
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isNewChat, setIsNewChat] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const { user } = useAuth();
  
  // Add a ref to track current chat ID during operations
  const currentChatIdRef = useRef(null);

  // Update ref when currentChat changes
  useEffect(() => {
    currentChatIdRef.current = currentChat?._id || null;
  }, [currentChat]);

  const cleanMessages = (msgs = []) => {
    return msgs
      .filter(msg => msg && msg.content && msg.role)
      .map(msg => ({
        ...msg,
        isTemp: false,
        id: msg.id?.replace(/^temp-/, '') || `${Date.now()}-${msg.role}`,
        content: msg.content.replace('(sending...)', '').trim(),
      }));
  };

  // Load chat history - IMPROVED VERSION
  const loadChatHistory = async (preserveCurrentChat = true) => {
    setIsLoadingHistory(true);
    try {
      const response = await chatAPI.getHistory();
      const history = Array.isArray(response.data) ? response.data : [];
      const sortedHistory = history.sort(
        (a, b) =>
          new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at)
      );

      const cleanedHistory = sortedHistory.map(chat => ({
        ...chat,
        messages: cleanMessages(chat.messages)
      }));

      setChatHistory(cleanedHistory);

      // Preserve current chat if it still exists
      if (preserveCurrentChat && currentChatIdRef.current) {
        const updatedCurrentChat = cleanedHistory.find(c => c._id === currentChatIdRef.current);
        if (updatedCurrentChat) {
          setCurrentChat(updatedCurrentChat);
          setIsNewChat(false);
        } else {
          // Current chat was deleted, reset to new chat
          setCurrentChat(null);
          setIsNewChat(true);
          setMessages([]);
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      setChatHistory([]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Load chat history on mount
  useEffect(() => {
    if (user) loadChatHistory();
  }, [user]);

  // Update message display when chat changes
  useEffect(() => {
    if (currentChat && currentChat.messages) {
      setMessages(cleanMessages(currentChat.messages));
      setIsNewChat(false);
    } else {
      setMessages([]);
    }
  }, [currentChat]);

  // Handle sending new messages
  const handleNewMessage = async (newMessage, chatId = null) => {
    const newMsgs = newMessage.messages || [];
    
    if (newMsgs.length === 0) return;

    const timestamp = new Date().toISOString();
    const cleanNewMessages = newMsgs.map(msg => ({
      ...msg,
      content: msg.content.replace('(sending...)', '').trim()
    })).filter(msg => msg.content.length > 0);

    if (cleanNewMessages.length === 0) return;

    if (currentChat && !isNewChat) {
      const existingMessageIds = new Set(messages.map(msg => msg.id));
      const uniqueNewMessages = cleanNewMessages.filter(msg => !existingMessageIds.has(msg.id));
      
      if (uniqueNewMessages.length === 0) return;

      const updatedMessages = [...messages, ...uniqueNewMessages];
      const updatedChat = {
        ...currentChat,
        messages: updatedMessages,
        updated_at: timestamp,
      };
      
      setCurrentChat(updatedChat);
      setMessages(updatedMessages);

      setChatHistory(prev => 
        prev.map(chat => 
          chat._id === currentChat._id ? updatedChat : chat
        )
      );
    } else {
      const newChatId = chatId || newMessage.chat_id || `chat-${Date.now()}`;
      const newChat = {
        _id: newChatId,
        messages: cleanNewMessages,
        created_at: timestamp,
        updated_at: timestamp,
        title: cleanNewMessages[0]?.content?.substring(0, 30) + (cleanNewMessages[0]?.content?.length > 30 ? '...' : '') || 'New Chat',
      };
      
      setChatHistory(prev => [newChat, ...prev]);
      setCurrentChat(newChat);
      setMessages(cleanNewMessages);
      setIsNewChat(false);
    }
  };

  // Select existing chat
  const handleSelectChat = (chat) => {
    if (isLoadingHistory) return;
    
    const cleanChat = {
      ...chat,
      messages: (chat.messages || []).filter(msg => 
        msg.content && !msg.content.includes('(sending...)')
      )
    };
    
    setCurrentChat(cleanChat);
    setIsNewChat(false);
  };

  // Delete a single chat - FIXED VERSION
  const handleDeleteChat = async (chatToDelete) => {
    const wasCurrentChat = currentChat && currentChat._id === chatToDelete._id;
    
    try {
      if (chatToDelete._id) await chatAPI.deleteChat(chatToDelete._id);
      
      // Only reload history, don't reset current chat unless it's the one being deleted
      await loadChatHistory(!wasCurrentChat);
      
      if (wasCurrentChat) {
        // If we deleted the current chat, reset to new chat
        setCurrentChat(null);
        setIsNewChat(true);
        setMessages([]);
      }
      // If we deleted another chat, current chat is preserved by loadChatHistory
    } catch (error) {
      console.error('Error deleting chat:', error);
      await loadChatHistory(!wasCurrentChat);
    }
  };

  // Clear all chats
  const handleClearAll = async () => {
    try {
      await chatAPI.clearAllChats();
      setChatHistory([]);
      setCurrentChat(null);
      setMessages([]);
      setIsNewChat(true);
    } catch (error) {
      console.error('Error clearing all chats:', error);
      setChatHistory([]);
    }
  };

  const handleNewChat = () => {
    setCurrentChat(null);
    setMessages([]);
    setIsNewChat(true);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Sidebar
        chatHistory={chatHistory}
        onSelectChat={handleSelectChat}
        currentChat={currentChat}
        onDeleteChat={handleDeleteChat}
        onClearAll={handleClearAll}
        onNewChat={handleNewChat}
        isNewChat={isNewChat}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        isLoading={isLoadingHistory}
      />

      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          sidebarOpen ? 'ml-80' : 'ml-0'
        }`}
      >
        <ChatWindow
          messages={messages}
          onNewMessage={handleNewMessage}
          currentChat={currentChat}
          isNewChat={isNewChat}
          onClearAll={handleClearAll}
          isLoadingHistory={isLoadingHistory}
        />
      </div>
    </div>
  );
};

export default Chat;