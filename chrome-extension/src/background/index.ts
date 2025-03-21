import 'webextension-polyfill';
import { exampleThemeStorage } from '@extension/storage';

exampleThemeStorage.get().then(theme => {
  console.log('theme', theme);
});

console.log('Background loaded');
console.log("Edit 'chrome-extension/src/background/index.ts' and save to reload.");

// ADDED: Background script to open side panel on click
chrome.action.onClicked.addListener(tab => {
  if (!tab.id) return;

  chrome.sidePanel.setOptions({
    tabId: tab.id,
    path: 'side-panel/index.html',
    enabled: true,
  });

  chrome.sidePanel.open({ tabId: tab.id });
});
