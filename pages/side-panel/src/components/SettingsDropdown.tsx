import React, { useState } from 'react';
import { ToggleButton } from '@extension/ui';
import { getSupabaseClient } from '@extension/shared/lib/utils/supabaseClient';

interface SettingsDropdownProps {
  theme: 'light' | 'dark';
}

const GearIcon: React.FC<React.SVGProps<SVGSVGElement>> = props => {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-6 h-6"
      {...props}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .7.41 1.3 1.01 1.57.2.1.42.16.65.18H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
};

const SettingsDropdown: React.FC<SettingsDropdownProps> = ({ theme }) => {
  const [isOpen, setIsOpen] = useState(false);
  const supabase = getSupabaseClient();

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.error('Error signing out:', error);
        return;
      }
      await chrome.storage.local.remove('session');
      window.location.reload();
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`p-2 rounded-full hover:bg-opacity-10 hover:bg-gray-500 transition-colors`}>
        <GearIcon />
      </button>

      {isOpen && (
        <div
          className={`absolute right-0 mt-2 w-48 rounded-md shadow-lg py-1 ${
            theme === 'light' ? 'bg-white' : 'bg-[#2A2A2A]'
          } ring-1 ring-black ring-opacity-5`}>
          <div
            className={`px-4 py-2 ${theme === 'light' ? 'text-gray-700 hover:bg-gray-100' : 'text-gray-300 hover:bg-[#3A3A3A]'}`}>
            <ToggleButton />
          </div>
          <button
            onClick={handleLogout}
            className={`w-full text-left px-4 py-2 text-sm ${
              theme === 'light' ? 'text-gray-700 hover:bg-gray-100' : 'text-gray-300 hover:bg-[#3A3A3A]'
            }`}>
            Log out
          </button>
        </div>
      )}
    </div>
  );
};

export default SettingsDropdown;
