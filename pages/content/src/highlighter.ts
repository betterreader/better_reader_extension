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

export function initHighlighter() {
  let currentSelection: Selection | null = null;
  let popup: HTMLDivElement | null = null;
  let isClickingPopup = false;

  function createHighlightPopup(x: number, y: number) {
    if (popup) {
      document.body.removeChild(popup);
    }

    const newPopup = document.createElement('div');
    newPopup.className = 'highlight-popup';
    newPopup.style.position = 'absolute';
    newPopup.style.zIndex = '9999';
    newPopup.style.backgroundColor = 'white';
    newPopup.style.border = '1px solid #ccc';
    newPopup.style.borderRadius = '4px';
    newPopup.style.padding = '8px';
    newPopup.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.2)';
    newPopup.style.left = `${x}px`;
    newPopup.style.top = `${y - 45}px`;

    // Create container for color buttons
    const colorContainer = document.createElement('div');
    colorContainer.className = 'highlight-color-container';
    colorContainer.style.display = 'flex';
    colorContainer.style.marginBottom = '8px';
    colorContainer.style.justifyContent = 'space-between';

    const colors = ['yellow', 'green', 'blue', 'pink'];
    colors.forEach(color => {
      const button = document.createElement('button');
      button.className = 'highlight-color-btn';
      button.style.width = '24px';
      button.style.height = '24px';
      button.style.borderRadius = '50%';
      button.style.border = '1px solid #ddd';
      button.style.margin = '0 4px';
      button.style.cursor = 'pointer';
      button.style.backgroundColor = color;

      // Use mousedown instead of click to ensure it fires before popup removal
      button.addEventListener('mousedown', e => {
        e.preventDefault();
        e.stopPropagation();
        isClickingPopup = true;
        console.log('Color button mousedown:', color);
        dispatchHighlightEvent(color);
      });

      colorContainer.appendChild(button);
    });

    newPopup.appendChild(colorContainer);

    // Add a separator
    const separator = document.createElement('div');
    separator.style.borderTop = '1px solid #eee';
    separator.style.margin = '4px 0';
    newPopup.appendChild(separator);

    // Add ELI5 button
    const eli5Button = document.createElement('button');
    eli5Button.className = 'eli5-button';
    eli5Button.textContent = 'ELI5';
    eli5Button.style.backgroundColor = '#4285f4';
    eli5Button.style.color = 'white';
    eli5Button.style.border = 'none';
    eli5Button.style.borderRadius = '4px';
    eli5Button.style.padding = '4px 8px';
    eli5Button.style.fontSize = '12px';
    eli5Button.style.fontWeight = 'bold';
    eli5Button.style.cursor = 'pointer';
    eli5Button.style.width = '100%';
    eli5Button.style.textAlign = 'center';

    eli5Button.addEventListener('mousedown', e => {
      e.preventDefault();
      e.stopPropagation();
      isClickingPopup = true;

      // Get the current selection
      if (currentSelection && !currentSelection.isCollapsed) {
        const selectedText = currentSelection.toString();

        // Send message to background script to trigger explanation with "examples" mode
        chrome.runtime.sendMessage({
          action: 'explainWithAI',
          text: selectedText,
          mode: 'examples',
          source: 'eli5Button',
        });

        // Open side panel
        chrome.runtime.sendMessage({
          action: 'openSidePanel',
        });
      }

      // Remove popup after clicking
      if (popup) {
        document.body.removeChild(popup);
        popup = null;
      }
    });

    newPopup.appendChild(eli5Button);

    document.body.appendChild(newPopup);
    popup = newPopup;
  }

  function dispatchHighlightEvent(color: string) {
    console.log('Dispatching highlight event with color:', color);

    if (!currentSelection || currentSelection.isCollapsed) {
      console.log('No valid selection');
      return;
    }

    try {
      const range = currentSelection.getRangeAt(0);
      console.log('Range:', {
        startContainer: range.startContainer,
        endContainer: range.endContainer,
        startOffset: range.startOffset,
        endOffset: range.endOffset,
        text: range.toString(),
      });

      // Create a serializable range object
      const rangeData = {
        startOffset: range.startOffset,
        endOffset: range.endOffset,
        startContainer: range.startContainer,
        endContainer: range.endContainer,
      };

      // Dispatch event for content-runtime to handle the actual highlighting
      const event = new CustomEvent<HighlightEvent>('HIGHLIGHT_TEXT', {
        bubbles: true,
        composed: true,
        detail: {
          type: 'HIGHLIGHT_TEXT',
          color,
          selection: {
            text: range.toString(),
            range: rangeData,
          },
        },
      });

      console.log('Dispatching event:', event);
      window.dispatchEvent(event);
      console.log('Successfully dispatched highlight event');
    } catch (e) {
      console.error('Failed to dispatch highlight event:', e);
    }

    // Use setTimeout to ensure the event is dispatched before cleanup
    setTimeout(() => {
      if (popup) {
        document.body.removeChild(popup);
        popup = null;
      }

      // Clear the selection
      if (currentSelection) {
        currentSelection.removeAllRanges();
        currentSelection = null;
      }
    }, 0);
  }

  // Handle text selection
  document.addEventListener('mouseup', e => {
    // Reset clicking state
    if (isClickingPopup) {
      isClickingPopup = false;
      return;
    }

    // Don't show popup if clicking on the popup itself
    if (popup && (e.target as Node).contains(popup)) {
      return;
    }

    const selection = window.getSelection();
    console.log('Selection event:', {
      selection,
      isCollapsed: selection?.isCollapsed,
      type: selection?.type,
      text: selection?.toString(),
    });

    if (!selection || selection.isCollapsed) {
      if (popup) {
        document.body.removeChild(popup);
        popup = null;
      }
      return;
    }

    const selectedText = selection.toString().trim();
    if (selectedText) {
      currentSelection = selection;
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();

      // Position popup above the selection
      const x = rect.left + rect.width / 2 - 50;
      const y = rect.top + window.scrollY;

      console.log('Creating popup at:', { x, y, rect });
      createHighlightPopup(x, y);
    }
  });

  // Clean up popup when clicking outside
  document.addEventListener('mousedown', e => {
    // Don't remove popup if clicking on it
    if (popup && popup.contains(e.target as Node)) {
      return;
    }

    // Only remove popup if we're not in the middle of a color selection
    if (popup && !isClickingPopup) {
      document.body.removeChild(popup);
      popup = null;
    }
  });

  // Prevent text selection from being lost when moving mouse over popup
  document.addEventListener('mousemove', e => {
    if (popup && popup.contains(e.target as Node)) {
      e.preventDefault();
    }
  });

  console.log('Highlighter initialized with event listeners');
}
