import { OpenAI } from 'openai';
import { NextResponse } from 'next/server';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: Request) {
  try {
    const { response, articleContent } = await request.json();

    if (!response || !articleContent) {
      return NextResponse.json({ error: 'Response and article content are required' }, { status: 400 });
    }

    const feedbackResponse = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content:
            'You are an expert at evaluating student responses to article questions. Provide constructive feedback that helps the student improve their understanding.',
        },
        {
          role: 'user',
          content: `Article content: ${articleContent}
          
Student response: ${response}

Provide feedback on this response, considering:
1. Understanding of the article's main points
2. Critical thinking and analysis
3. Areas for improvement
4. Suggestions for deeper exploration`,
        },
      ],
      max_tokens: 500,
    });

    const feedback = feedbackResponse.choices[0]?.message?.content || '';

    return NextResponse.json({ feedback });
  } catch (error) {
    console.error('Error generating feedback:', error);
    return NextResponse.json({ error: 'Failed to generate feedback' }, { status: 500 });
  }
}
