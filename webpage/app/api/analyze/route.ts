import { OpenAI } from 'openai';
import { NextResponse } from 'next/server';
import * as cheerio from 'cheerio';
import { createClient } from '@/utils/supabase/server';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function fetchArticleContent(url: string) {
  try {
    console.log('Fetching article content from:', url);
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);

    let content = $('article').text();
    if (!content.trim()) {
      content = $('main').text();
    }
    if (!content.trim()) {
      content = $('body').text();
    }

    return content.replace(/\s+/g, ' ').trim();
  } catch (error) {
    console.error('Error fetching article content:', error);
    throw new Error('Failed to fetch article content');
  }
}

export async function POST(request: Request) {
  try {
    const { url } = await request.json();

    if (!url) {
      return NextResponse.json({ error: 'URL is required' }, { status: 400 });
    }

    console.log('Calling OpenAI API to analyze article');

    const articleContent = await fetchArticleContent(url);

    // Get comprehensive analysis from OpenAI
    const analysisResponse = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are an expert at analyzing articles and creating educational content.',
        },
        {
          role: 'user',
          content: `Analyze this article and provide:
		  1. The title of the article
		  2. A concise summary
		  3. Essential context/background information
		  4. Key technical terms and their definitions
		  5. Multiple-choice questions
		  6. Short response reflection questions
		  7. Future research recommendations

		  Article: ${articleContent}`,
        },
      ],
      functions: [
        {
          name: 'provide_article_analysis',
          description: 'Provide comprehensive analysis of the article',
          parameters: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              summary: { type: 'string' },
              context: { type: 'string' },
              keyTerms: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    term: { type: 'string' },
                    definition: { type: 'string' },
                  },
                  required: ['term', 'definition'],
                },
              },
              questions: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    question: { type: 'string' },
                    options: {
                      type: 'array',
                      items: { type: 'string' },
                    },
                    correctAnswer: { type: 'number' },
                  },
                  required: ['question', 'options', 'correctAnswer'],
                },
              },
              shortResponseQuestions: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    question: { type: 'string' },
                  },
                  required: ['id', 'question'],
                },
              },
              futureResearch: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    topic: { type: 'string' },
                    description: { type: 'string' },
                  },
                  required: ['topic', 'description'],
                },
              },
            },
            required: ['summary', 'context', 'keyTerms', 'questions', 'shortResponseQuestions', 'futureResearch'],
          },
        },
      ],
      function_call: { name: 'provide_article_analysis' },
    });

    console.log('Saving analysis to Supabase');
    const functionArgs = analysisResponse.choices[0]?.message?.function_call?.arguments || '{}';
    const analysis = JSON.parse(functionArgs);

    const supabase = await createClient();
    // get user
    let {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      console.log('User not found');
      return NextResponse.json({ error: 'User not found' }, { status: 401 });
    }

    let { data, error } = await supabase
      .from('analysis')
      .insert([
        {
          user_id: user.id,
          title: analysis.title,
          url: url,
          article_content: articleContent,
          summary: analysis.summary,
          context: analysis.context,
          key_terms: analysis.keyTerms,
          questions: analysis.questions,
          short_response_questions: analysis.shortResponseQuestions,
          future_research: analysis.futureResearch,
        },
      ])
      .select();

    if (error || !data || data.length === 0) {
      console.error('Error saving analysis:', error);
      return NextResponse.json({ error: 'Failed to save analysis' }, { status: 500 });
    }

    console.log('Returning analysis result');
    return NextResponse.json({
      ...analysis,
      articleContent: 'Fake Content', // Include the original article content in the response
      analysisId: data[0].id,
    });
  } catch (error) {
    console.error('Error processing article:', error);
    return NextResponse.json({ error: 'Failed to process article' }, { status: 500 });
  }
}
