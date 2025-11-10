import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

const Sidebar = ({ 
  chatHistory, 
  onSelectChat, 
  currentChat, 
  onDeleteChat, 
  onClearAll,
  onNewChat,
  isNewChat,
  isOpen, 
  onToggle 
}) => {
  const { user, logout } = useAuth();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Handle deleting chat
  const handleDeleteChat = (chat, e) => {
    e.stopPropagation();
    onDeleteChat(chat);
  };

  // Confirm clear all
  const confirmClearAll = () => {
    onClearAll();
    setShowDeleteConfirm(false);
  };

  // Start new chat
  const handleNewChat = () => {
    onNewChat();
  };

  // ✅ Display text fix: Show title or last user query
  const getChatDisplayText = (chat) => {
    const latestMsg = chat.messages?.length
      ? chat.messages[chat.messages.length - 1].content
      : '';
    const displayText = chat.title || latestMsg || 'New Chat';
    return displayText.length > 40
      ? displayText.substring(0, 40) + '...'
      : displayText;
  };

  // ✅ Robust date formatter
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Recent Chats';

    const today = new Date();
    const yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // ✅ Group chats by date (with fallbacks)
  const groupChatsByDate = () => {
    const groups = {};
    const sortedChats = [...chatHistory].sort(
      (a, b) =>
        new Date(b.updated_at || b.created_at) -
        new Date(a.updated_at || a.created_at)
    );

    sortedChats.forEach(chat => {
      const date = formatDate(chat.updated_at || chat.created_at);
      if (!groups[date]) groups[date] = [];
      groups[date].push(chat);
    });
    return groups;
  };

  const groupedChats = groupChatsByDate();

  // Sidebar collapsed view
  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="fixed left-4 top-4 z-50 bg-black text-white p-2 rounded-lg shadow-lg hover:bg-gray-800 transition-colors"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
    );
  }

  return (
    <div className="w-64 bg-gray-900 text-white flex flex-col h-full fixed left-0 top-0 z-40 border-r border-gray-700">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
              <span className="text-sm font-bold">HR</span>
            </div>
            <span className="font-semibold text-lg">HR Assistant</span>
          </div>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-white p-1 rounded transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New Chat Button */}
        <button
          onClick={handleNewChat}
          className={`w-full py-2.5 px-3 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
            isNewChat 
              ? 'bg-gray-700 text-white' 
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
          }`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New chat
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          {Object.keys(groupedChats).map((date) => (
            <div key={date} className="mb-6">
              <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">
                {date}
              </h3>
              <div className="space-y-1">
                {groupedChats[date].map((chat) => (
                  <div
                    key={chat._id}
                    onClick={() => onSelectChat(chat)}
                    className={`group relative p-2 rounded-lg cursor-pointer transition-colors ${
                      currentChat?._id === chat._id
                        ? 'bg-gray-700 text-white'
                        : 'text-gray-400 hover:bg-gray-800 hover:text-gray-300'
                    }`}
                  >
                    <div className="pr-6">
                      <div className="text-sm font-normal mb-1 leading-tight">
                        {getChatDisplayText(chat)}
                      </div>
                    </div>

                    {/* Delete Button */}
                    <button
                      onClick={(e) => handleDeleteChat(chat, e)}
                      className="absolute right-2 top-1/2 transform -translate-y-1/2 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 p-1 rounded transition-colors"
                      title="Delete this chat"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* No chats message */}
          {chatHistory.length === 0 && (
            <div className="text-center py-8">
              <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
                <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <p className="text-gray-500 text-sm">No chat history yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Clear All */}
      {chatHistory.length > 0 && (
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="w-full text-gray-400 hover:text-red-400 text-sm font-medium transition-colors py-2"
          >
            Clear all conversations
          </button>
        </div>
      )}

      {/* Clear All Modal */}
      {showDeleteConfirm && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-lg p-4 max-w-sm w-full">
            <h3 className="text-lg font-medium text-white mb-2">Clear all chats?</h3>
            <p className="text-gray-400 text-sm mb-4">
              This will permanently remove all your conversation history.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 bg-gray-700 text-white py-2 px-4 rounded text-sm font-medium hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmClearAll}
                className="flex-1 bg-red-500 text-white py-2 px-4 rounded text-sm font-medium hover:bg-red-600 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center gap-3 p-3 bg-gray-800 rounded-lg">
          <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
            {user?.username?.charAt(0).toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">{user?.username}</p>
          </div>
          <button
            onClick={logout}
            className="text-gray-400 hover:text-white p-1 rounded transition-colors"
            title="Logout"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
