import { CreateHighlightRequest, HighlightData, UpdateHighlightRequest } from '../types/highlight.js';

const API_BASE = 'http://localhost:5007/api';

export class HighlightService {
  private authToken: string;

  constructor(authToken: string) {
    this.authToken = authToken;
  }

  private async fetchWithAuth(url: string, options: RequestInit = {}) {
    const headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.authToken}`,
      ...options.headers,
    };

    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'API request failed');
    }
    return response.json();
  }

  async createHighlight(data: CreateHighlightRequest): Promise<HighlightData> {
    return this.fetchWithAuth(`${API_BASE}/highlights`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getHighlights(url: string): Promise<HighlightData[]> {
    return this.fetchWithAuth(`${API_BASE}/highlights?url=${encodeURIComponent(url)}`);
  }

  async updateHighlight(id: number, data: UpdateHighlightRequest): Promise<HighlightData> {
    return this.fetchWithAuth(`${API_BASE}/highlights/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteHighlight(id: number): Promise<void> {
    await this.fetchWithAuth(`${API_BASE}/highlights/${id}`, {
      method: 'DELETE',
    });
  }
}
