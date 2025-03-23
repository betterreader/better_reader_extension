import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

interface FutureResearchProps {
  recommendations: Array<{
    topic: string;
    description: string;
  }>;
}

export default function FutureResearch({ recommendations }: FutureResearchProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Future Research Topics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {recommendations.map((rec, index) => (
            <div key={index} className="border-b border-gray-200 dark:border-gray-700 last:border-0 pb-4 last:pb-0">
              <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-2">{rec.topic}</h3>
              <p className="text-gray-700 dark:text-gray-300">{rec.description}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
