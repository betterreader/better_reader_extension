import React, { useEffect, useRef, useState } from 'react';
import './highlight-styles.css';
import { CreateHighlightRequest, HighlightData, HighlightService, HighlightEvent } from '@extension/shared';
import { getSupabaseClient } from '@extension/shared/lib/utils/supabaseClient';
import { Session } from '@supabase/supabase-js';

interface SerializedRange {
  startXPath: string;
  startOffset: number;
  endXPath: string;
  endOffset: number;
}

const getXPath = (node: Node): string => {
  console.log('Getting XPath for node:', node);
  if (node.nodeType !== Node.ELEMENT_NODE && node.parentNode) {
    const parentPath = getXPath(node.parentNode);
    const siblings = Array.from(node.parentNode.childNodes).filter(n => n.nodeType === node.nodeType) as ChildNode[];
    const index = siblings.indexOf(node as ChildNode) + 1;
    const path = `${parentPath}/text()[${index}]`;
    console.log('Generated XPath:', path);
    return path;
  }
  if (!node.parentNode) {
    return '/';
  }
  const siblings = Array.from(node.parentNode.childNodes).filter(n => n.nodeName === node.nodeName) as ChildNode[];
  const index = siblings.indexOf(node as ChildNode) + 1;
  const path = `${getXPath(node.parentNode)}/${node.nodeName.toLowerCase()}[${index}]`;
  console.log('Generated XPath:', path);
  return path;
};

const getNodeByXPath = (xpath: string): Node | null => {
  console.log('Attempting to find node for XPath:', xpath);
  const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
  console.log('XPath evaluation result:', result.singleNodeValue);
  return result.singleNodeValue;
};

// Helper: given a container node (which might be an element) and an offset,
// return the first text node that contains the selection.
const getEffectiveTextNode = (container: Node, offset: number): { node: Text; offset: number } => {
  if (container.nodeType === Node.TEXT_NODE) {
    return { node: container as Text, offset };
  }
  // If container is an element, try to use its child at the given offset.
  const child = container.childNodes[offset];
  if (child) {
    if (child.nodeType === Node.TEXT_NODE) {
      return { node: child as Text, offset: 0 };
    } else {
      // search recursively for a text node inside the child
      const walker = document.createTreeWalker(child, NodeFilter.SHOW_TEXT, null);
      const firstText = walker.nextNode();
      if (firstText && firstText.nodeType === Node.TEXT_NODE) {
        return { node: firstText as Text, offset: 0 };
      }
    }
  }
  // fallback: use a TreeWalker on the container
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
  const firstText = walker.nextNode();
  if (firstText && firstText.nodeType === Node.TEXT_NODE) {
    return { node: firstText as Text, offset: 0 };
  }
  throw new Error('No text node found in container.');
};

const getTextNodeFromXPath = (xpath: string, offset: number): { node: Text; offset: number } => {
  const node = getNodeByXPath(xpath);
  if (!node) {
    throw new Error('Node not found for XPath: ' + xpath);
  }
  if (node.nodeType === Node.TEXT_NODE) {
    return { node: node as Text, offset };
  }
  return getEffectiveTextNode(node, offset);
};

// Helper function to compare xpaths positions in the document
function compareXPaths(xpath1: string, xpath2: string): number {
  const node1 = document.evaluate(xpath1, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
  const node2 = document.evaluate(xpath2, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;

  if (!node1 || !node2) return 0;

  // compareDocumentPosition returns a bitmask
  const position = node1.compareDocumentPosition(node2);

  // Node.DOCUMENT_POSITION_FOLLOWING (4) means node2 follows node1
  if (position & Node.DOCUMENT_POSITION_FOLLOWING) {
    return -1; // node1 comes first
  }
  // Node.DOCUMENT_POSITION_PRECEDING (2) means node2 precedes node1
  if (position & Node.DOCUMENT_POSITION_PRECEDING) {
    return 1; // node2 comes first
  }
  return 0;
}

export const Highlighter: React.FC = () => {
  console.log('Rendering content-runtime');
  const highlightsRef = useRef<{ [key: string]: HighlightData }>({});
  const currentUrl = window.location.href;

  const createHighlightElement = (color: string, id: string): HTMLSpanElement => {
    console.log('Creating highlight element with color:', color, 'and id:', id);
    const span = document.createElement('span');
    span.className = `highlight-${color}`;
    span.dataset.highlightId = id;
    return span;
  };

  const wrapPartialTextNode = (node: Text, startOffset: number, endOffset: number, color: string, id: string) => {
    const text = node.textContent || '';
    const beforeText = text.slice(0, startOffset);
    const highlightText = text.slice(startOffset, endOffset);
    const afterText = text.slice(endOffset);

    const parent = node.parentNode;
    if (!parent) return;

    if (beforeText) {
      parent.insertBefore(document.createTextNode(beforeText), node);
    }

    const span = createHighlightElement(color, id);
    span.appendChild(document.createTextNode(highlightText));
    parent.insertBefore(span, node);

    if (afterText) {
      parent.insertBefore(document.createTextNode(afterText), node);
    }

    parent.removeChild(node);
  };

  const wrapWholeTextNode = (node: Text, color: string, id: string) => {
    const span = createHighlightElement(color, id);
    const clonedNode = node.cloneNode(true);
    span.appendChild(clonedNode);
    node.parentNode?.replaceChild(span, node);
  };

  const highlightRange = (serializedRange: SerializedRange, color: string, id: string): boolean => {
    console.log('Starting highlight range process with:', { serializedRange, color, id });
    try {
      // Resolve effective text nodes even if the stored XPath points to an element.
      const { node: startTextNode, offset: startOffset } = getTextNodeFromXPath(
        serializedRange.startXPath,
        serializedRange.startOffset,
      );
      const { node: endTextNode, offset: endOffset } = getTextNodeFromXPath(
        serializedRange.endXPath,
        serializedRange.endOffset,
      );
      console.log('Effective text nodes found:', { startTextNode, endTextNode });

      if (startTextNode === endTextNode) {
        wrapPartialTextNode(startTextNode, startOffset, endOffset, color, id);
      } else {
        wrapPartialTextNode(startTextNode, startOffset, startTextNode.textContent?.length || 0, color, id);

        const range = document.createRange();
        range.setStart(startTextNode, startOffset);
        range.setEnd(endTextNode, endOffset);

        const walker = document.createTreeWalker(range.commonAncestorContainer, NodeFilter.SHOW_TEXT, {
          acceptNode: (node: Node) => {
            if (
              node.nodeType === Node.TEXT_NODE &&
              node !== startTextNode &&
              node !== endTextNode &&
              range.intersectsNode(node)
            ) {
              return NodeFilter.FILTER_ACCEPT;
            }
            return NodeFilter.FILTER_REJECT;
          },
        });

        let node: Node | null;
        while ((node = walker.nextNode())) {
          if (node instanceof Text) {
            wrapWholeTextNode(node, color, id);
          }
        }

        wrapPartialTextNode(endTextNode, 0, endOffset, color, id);
      }

      document.body.normalize();
      console.log('Successfully applied highlight:', { id, color });
      return true;
    } catch (e) {
      console.error('Highlight range error:', e);
      return false;
    }
  };

  const handleNewHighlight = async (event: Event) => {
    const customEvent = event as CustomEvent<HighlightEvent>;
    const { color, selection } = customEvent.detail;
    console.log('Received highlight event:', customEvent.detail);

    const id = Math.random().toString(36).substr(2, 9);
    const serializedRange: SerializedRange = {
      startXPath: getXPath(selection.range.startContainer),
      startOffset: selection.range.startOffset,
      endXPath: getXPath(selection.range.endContainer),
      endOffset: selection.range.endOffset,
    };
    const start_xpath = getXPath(selection.range.startContainer);
    const start_offset = selection.range.startOffset;
    const end_xpath = getXPath(selection.range.endContainer);
    const end_offset = selection.range.endOffset;

    const highlightData: CreateHighlightRequest = {
      color,
      start_offset,
      end_offset,
      start_xpath,
      end_xpath,
      text: selection.text,
      url: currentUrl,
      comment: '',
      local_id: id,
      article_title: document.title,
    };
    if (highlightRange(serializedRange, color, id)) {
      // Instead of saving highlights, we can just dispatch a new event to the side panel
      // await saveHighlight(highlightData);
      chrome.runtime.sendMessage({
        type: 'HIGHLIGHT_TEXT_RUNTIME',
        ...highlightData,
      });
      console.log('Content-Runtime: Successfully sent highlight event:', event);
    }
  };

  const handleDeleteHighlight = (highlightId: string) => {
    const elements = Array.from(document.querySelectorAll(`[data-highlight-id="${highlightId}"]`)) as HTMLElement[];
    elements.forEach(element => {
      const text = element.textContent || '';
      if (element.parentElement) {
        const textNode = document.createTextNode(text);
        element.parentElement.insertBefore(textNode, element);
        element.remove();
      }
    });
    document.body.normalize();
  };

  const handleScrollToHighlight = (highlightId: string) => {
    const element = document.querySelector(`[data-highlight-id="${highlightId}"]`) as HTMLElement | null;
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('highlight-focus');
      setTimeout(() => element.classList.remove('highlight-focus'), 2000);
    }
  };

  const loadHighlights = async () => {
    console.log('Content-Runtime: Starting to load highlights for URL:', currentUrl);
    if (!session || !session.access_token) {
      console.error('Content-Runtime: No session found in storage for loading highlights');
      return;
    }
    const { data, error } = await supabase
      .from('highlight')
      .select()
      .eq('url', currentUrl)
      .eq('user_id', session.user.id)
      .select();
    if (error) {
      console.error('Error loading highlights:', error);
      return;
    }
    console.log('Content-Runtime: Retrieved highlights:', data);

    Object.values(data).forEach(highlight => {
      console.log('Applying highlight:', highlight);
      const serializedRange: SerializedRange = {
        startXPath: highlight.start_xpath,
        startOffset: highlight.start_offset,
        endXPath: highlight.end_xpath,
        endOffset: highlight.end_offset,
      };
      highlightRange(serializedRange, highlight.color, highlight.local_id);
      highlightsRef.current[highlight.local_id] = highlight;
    });
    console.log('Highlights loaded and applied.');
  };

  const [session, setSession] = useState<Session | null>(null);
  const supabase = getSupabaseClient();

  async function getSessionFromStorage() {
    try {
      const { session } = (await chrome.storage.local.get('session')) as { session: Session | null };
      if (session) {
        const { error: supaAuthError } = await supabase.auth.setSession(session);
        if (supaAuthError) {
          setSession(null);
          throw supaAuthError;
        }
        console.log('Content-Runtime: Getting session from storage', session);
        setSession(session);
      }
    } catch (error) {
      console.error('Error getting session:', error);
      setSession(null);
    }
  }

  // Get session on mount
  useEffect(() => {
    getSessionFromStorage();
  }, []);

  // Initialize event listeners and load highlights only after a valid session is available
  useEffect(() => {
    if (!session || !session.access_token) {
      console.error('Content-Runtime: No valid session available yet.');
      return;
    }

    console.log('Initializing Highlighter component with valid session');
    window.addEventListener('HIGHLIGHT_TEXT', handleNewHighlight);

    const messageListener = (message: any) => {
      if (message.type === 'DELETE_HIGHLIGHT') {
        console.log('Content-Runtime: Received delete highlight message:', message);
        handleDeleteHighlight(message.local_id);
      } else if (message.type === 'SCROLL_TO_HIGHLIGHT') {
        console.log('Content-Runtime: Received scroll to highlight message:', message);
        handleScrollToHighlight(message.local_id);
      } else if (message.type === 'SESSION_UPDATED' && 'session' in message) {
        supabase.auth
          .setSession(message.session)
          .then(({ data: { session } }) => {
            setSession(session);
          })
          .catch(error => {
            console.error('Error setting session:', error);
          });
      }
    };

    chrome.runtime.onMessage.addListener(messageListener);

    // Load highlights now that we have a session
    loadHighlights();

    return () => {
      console.log('Cleaning up Highlighter component');
      window.removeEventListener('HIGHLIGHT_TEXT', handleNewHighlight);
      chrome.runtime.onMessage.removeListener(messageListener);
    };
  }, [session]);

  return null;
};
