import React, { useEffect, useRef } from 'react';
import './highlight-styles.css';

interface SerializedRange {
  startXPath: string;
  startOffset: number;
  endXPath: string;
  endOffset: number;
}

interface HighlightEvent {
  type: 'HIGHLIGHT_TEXT';
  color: string;
  selection: {
    text: string;
    range: {
      startOffset: number;
      endOffset: number;
      startContainer: Node;
      endContainer: Node;
    };
  };
}

interface HighlightData {
  id: string;
  color: string;
  text: string;
  timestamp: number;
  range: SerializedRange;
  comment?: string;
}

interface StorageData {
  [url: string]: {
    highlights: { [key: string]: HighlightData };
  };
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

const storage = {
  async get(key: string): Promise<StorageData> {
    return new Promise(resolve => {
      chrome.storage.local.get(key, result => resolve(result as StorageData));
    });
  },
  async set(items: StorageData): Promise<void> {
    return new Promise<void>(resolve => {
      chrome.storage.local.set(items, () => resolve());
    });
  },
};

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

  const saveHighlight = async (highlight: HighlightData) => {
    console.log('Attempting to save highlight:', highlight);
    try {
      const result = await storage.get(currentUrl);
      console.log('Current stored highlights:', result);
      const currentHighlights = (result[currentUrl] && result[currentUrl].highlights) || {};
      await storage.set({
        [currentUrl]: {
          highlights: {
            ...currentHighlights,
            [highlight.id]: highlight,
          },
        },
      });
      console.log('Successfully saved highlight:', highlight);
    } catch (e) {
      console.error('Save highlight error:', e);
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

    const highlightData: HighlightData = {
      id,
      color,
      text: selection.text,
      timestamp: Date.now(),
      range: serializedRange,
    };

    if (highlightRange(serializedRange, color, id)) {
      highlightsRef.current[id] = highlightData;
      await saveHighlight(highlightData);
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
    console.log('Starting to load highlights for URL:', currentUrl);
    try {
      const result = await storage.get(currentUrl);
      console.log('Retrieved stored highlights:', result);
      const storedHighlights = (result[currentUrl] && result[currentUrl].highlights) || {};
      console.log('Processing stored highlights:', storedHighlights);

      Object.values(storedHighlights).forEach(highlight => {
        console.log('Applying highlight:', highlight);
        highlightRange(highlight.range, highlight.color, highlight.id);
        highlightsRef.current[highlight.id] = highlight;
      });
      console.log('Highlights loaded and applied.');
    } catch (e) {
      console.error('Failed to load highlights:', e);
    }
  };

  useEffect(() => {
    console.log('Initializing Highlighter component');
    window.addEventListener('HIGHLIGHT_TEXT', handleNewHighlight);
    loadHighlights();

    const messageListener = (message: any) => {
      if (message.type === 'DELETE_HIGHLIGHT') {
        handleDeleteHighlight(message.highlightId);
      } else if (message.type === 'SCROLL_TO_HIGHLIGHT') {
        handleScrollToHighlight(message.highlightId);
      }
    };

    chrome.runtime.onMessage.addListener(messageListener);

    return () => {
      console.log('Cleaning up Highlighter component');
      window.removeEventListener('HIGHLIGHT_TEXT', handleNewHighlight);
      chrome.runtime.onMessage.removeListener(messageListener);
    };
  }, []);

  return null;
};
