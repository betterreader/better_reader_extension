// TODO: move this to a shared component
export const renderMarkdownText = (text: string): React.ReactNode[] => {
  if (!text) return [];

  const segments: React.ReactNode[] = [];
  let currentIndex = 0;

  // This regex captures:
  // 1-2: Bold: **text** or __text__
  // 3-4: Italic: *text* or _text_
  // 5-6: Inline code: `code`
  // 7-8: Link: [text](url)
  // 9-10: Strikethrough: ~~text~~
  // 11-13: Image: ![alt](url)
  const markdownRegex =
    /(?:(\*\*|__)(.*?)\1)|(?:(\*|_)(.*?)\3)|(?:(`)(.*?)\5)|(?:\[([^\]]+)\]\(([^)]+)\))|(?:(~~)(.*?)\8)|(?:(!\[)([^\]]*)\]\(([^)]+)\))/g;
  let lastIndex = 0;
  let match;

  while ((match = markdownRegex.exec(text)) !== null) {
    // Add any text before the matched markdown
    if (match.index > lastIndex) {
      const beforeText = text.slice(lastIndex, match.index);
      segments.push(...processNewlines(beforeText));
    }

    if (match[1]) {
      // Bold text
      segments.push(
        <span key={`bold-${currentIndex++}`} className="font-bold">
          {match[2]}
        </span>,
      );
    } else if (match[3]) {
      // Italic text
      segments.push(
        <span key={`italic-${currentIndex++}`} className="italic">
          {match[4]}
        </span>,
      );
    } else if (match[5]) {
      // Inline code
      segments.push(
        <code key={`code-${currentIndex++}`} className="bg-gray-100 p-1 rounded">
          {match[6]}
        </code>,
      );
    } else if (match[7]) {
      // Link
      segments.push(
        <a key={`link-${currentIndex++}`} href={match[8]} className="text-blue-500 underline">
          {match[7]}
        </a>,
      );
    } else if (match[9]) {
      // Strikethrough
      segments.push(
        <span key={`strike-${currentIndex++}`} className="line-through">
          {match[10]}
        </span>,
      );
    } else if (match[11]) {
      // Image
      segments.push(<img key={`img-${currentIndex++}`} src={match[13]} alt={match[12]} className="inline" />);
    }

    lastIndex = match.index + match[0].length;
  }

  // Process any remaining text after the last markdown match
  if (lastIndex < text.length) {
    segments.push(...processNewlines(text.slice(lastIndex)));
  }

  return segments;
};

// Helper function to process newlines in text
export const processNewlines = (text: string): React.ReactNode[] => {
  return text.split('\n').reduce((acc: React.ReactNode[], line, i) => {
    if (i > 0) acc.push(<br key={`br-${i}`} />);
    if (line) acc.push(line);
    return acc;
  }, []);
};
