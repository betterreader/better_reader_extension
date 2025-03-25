// TODO: fix null types
export interface HighlightData {
  id: number;
  created_at: string;
  url: string;
  color: string;
  start_xpath: string;
  start_offset: number;
  end_xpath: string;
  end_offset: number;
  comment: string;
  user_id: string;
  text: string;
  local_id: string;
  article_title: string;
}

export interface CreateHighlightRequest {
  url: string;
  color: string;
  start_xpath: string;
  start_offset: number;
  end_xpath: string;
  end_offset: number;
  text: string;
  comment: string;
  local_id: string;
  article_title: string;
}

export interface UpdateHighlightRequest {
  color?: string;
  comment?: string;
}

export interface HighlightEvent {
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

export interface HighlightEventRuntime extends CreateHighlightRequest {
  type: 'HIGHLIGHT_TEXT_RUNTIME';
}
