import React, { useState, useRef, useEffect } from 'react';
import type { ArticleData } from '../SidePanel';

interface SummaryTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

interface SummaryOptions {
  bulletPoints: {
    enabled: boolean;
    detail: 'concise' | 'detailed';
    level: 'beginner' | 'expert';
  };
  definitions: boolean;
  topics: boolean;
  questions: boolean;
}

interface SummaryResponse {
  bulletPoints?: string[];
  definitions?: { term: string; definition: string }[];
  topics?: string[];
  questions?: string[];
}

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  label: string;
  theme: 'light' | 'dark';
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ checked, onChange, label, theme }) => (
  <label className="flex items-center space-x-2 cursor-pointer">
    <span className={theme === 'light' ? 'text-gray-900' : 'text-gray-200'}>{label}</span>
    <div className="relative">
      <input type="checkbox" checked={checked} onChange={onChange} className="sr-only peer" />
      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-300 rounded-full dark:bg-gray-700 peer-checked:bg-blue-500 transition-colors duration-200 ease-in-out"></div>
      <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform duration-200 ease-in-out peer-checked:translate-x-5"></div>
    </div>
  </label>
);

const SummaryTab: React.FC<SummaryTabProps> = ({ articleData, apiBaseUrl, theme }) => {
  const summaryContainerRef = useRef<HTMLDivElement>(null);
  const [options, setOptions] = useState<SummaryOptions>({
    bulletPoints: {
      enabled: true,
      detail: 'concise',
      level: 'beginner',
    },
    definitions: false,
    topics: false,
    questions: false,
  });
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const scrollToBottom = () => {
    if (summaryContainerRef.current) {
      summaryContainerRef.current.scrollTop = summaryContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    if (summary) {
      setTimeout(scrollToBottom, 100); // Delay to ensure content is rendered
    }
  }, [summary]);

  const handleGenerateSummary = async () => {
    if (!articleData) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/generate_summary`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: articleData.content,
          options: options,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate summary');

      const data = await response.json();
      setSummary(data);
    } catch (error) {
      console.error('Error generating summary:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div
        ref={summaryContainerRef}
        className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 max-h-[calc(100vh-220px)]">
        <div className={`p-4 rounded-lg ${theme === 'light' ? 'bg-gray-100' : 'bg-[#2D2D2D]'}`}>
          <div className="mb-4">
            <ToggleSwitch
              checked={options.bulletPoints.enabled}
              onChange={e =>
                setOptions(prev => ({
                  ...prev,
                  bulletPoints: { ...prev.bulletPoints, enabled: e.target.checked },
                }))
              }
              label="Bullet Point Summary"
              theme={theme}
            />
            {options.bulletPoints.enabled && (
              <div className="ml-6 mt-2 space-y-2">
                <div className="flex space-x-4">
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      checked={options.bulletPoints.detail === 'concise'}
                      onChange={() =>
                        setOptions(prev => ({
                          ...prev,
                          bulletPoints: { ...prev.bulletPoints, detail: 'concise' },
                        }))
                      }
                    />
                    <span className={theme === 'light' ? 'text-gray-700' : 'text-gray-300'}>Concise</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      checked={options.bulletPoints.detail === 'detailed'}
                      onChange={() =>
                        setOptions(prev => ({
                          ...prev,
                          bulletPoints: { ...prev.bulletPoints, detail: 'detailed' },
                        }))
                      }
                    />
                    <span className={theme === 'light' ? 'text-gray-700' : 'text-gray-300'}>Detailed</span>
                  </label>
                </div>
                <div className="flex space-x-4">
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      checked={options.bulletPoints.level === 'beginner'}
                      onChange={() =>
                        setOptions(prev => ({
                          ...prev,
                          bulletPoints: { ...prev.bulletPoints, level: 'beginner' },
                        }))
                      }
                    />
                    <span className={theme === 'light' ? 'text-gray-700' : 'text-gray-300'}>Beginner</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      checked={options.bulletPoints.level === 'expert'}
                      onChange={() =>
                        setOptions(prev => ({
                          ...prev,
                          bulletPoints: { ...prev.bulletPoints, level: 'expert' },
                        }))
                      }
                    />
                    <span className={theme === 'light' ? 'text-gray-700' : 'text-gray-300'}>Expert</span>
                  </label>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <ToggleSwitch
              checked={options.definitions}
              onChange={e => setOptions(prev => ({ ...prev, definitions: e.target.checked }))}
              label="Key Definitions"
              theme={theme}
            />
            <ToggleSwitch
              checked={options.topics}
              onChange={e => setOptions(prev => ({ ...prev, topics: e.target.checked }))}
              label="Key Topics"
              theme={theme}
            />
            <ToggleSwitch
              checked={options.questions}
              onChange={e => setOptions(prev => ({ ...prev, questions: e.target.checked }))}
              label="Exploration Questions"
              theme={theme}
            />
          </div>
        </div>

        {summary && (
          <div className={`space-y-4 ${theme === 'light' ? 'text-gray-900' : 'text-gray-200'}`}>
            {summary.bulletPoints && summary.bulletPoints.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-semibold text-lg">Summary</h3>
                <ul className="list-disc pl-5 space-y-2">
                  {summary.bulletPoints.map((point, index) => (
                    <li key={index}>{point}</li>
                  ))}
                </ul>
              </div>
            )}

            {summary.definitions && summary.definitions.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-semibold text-lg">Key Definitions</h3>
                <div className="space-y-2">
                  {summary.definitions.map((def, index) => (
                    <div key={index} className="pl-5">
                      <span className="font-medium">{def.term}:</span> {def.definition}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {summary.topics && summary.topics.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-semibold text-lg">Key Topics</h3>
                <ul className="list-disc pl-5 space-y-1">
                  {summary.topics.map((topic, index) => (
                    <li key={index}>{topic}</li>
                  ))}
                </ul>
              </div>
            )}

            {summary.questions && summary.questions.length > 0 && (
              <div className="space-y-2">
                <h3 className="font-semibold text-lg">Exploration Questions</h3>
                <ul className="list-decimal pl-5 space-y-2">
                  {summary.questions.map((question, index) => (
                    <li key={index}>{question}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      <div
        className={`p-3 border-t sticky bottom-0 ${theme === 'light' ? 'border-gray-200 bg-white' : 'border-[#333] bg-[#1E1E1E]'}`}>
        <button
          onClick={handleGenerateSummary}
          disabled={loading || !articleData}
          className={`w-full py-2 px-4 rounded-md font-medium ${
            theme === 'light'
              ? 'bg-blue-500 text-white hover:bg-blue-600'
              : 'bg-[#297FFF] text-white hover:bg-[#2171E8]'
          } disabled:opacity-50 disabled:cursor-not-allowed`}>
          {loading ? 'Generating...' : 'Generate Summary'}
        </button>
      </div>
    </div>
  );
};

export default SummaryTab;
