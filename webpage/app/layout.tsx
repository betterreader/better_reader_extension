import HeaderAuth from '@/components/header-auth';
import { Geist } from 'next/font/google';
import { ThemeProvider } from 'next-themes';
import './globals.css';

const defaultUrl = process.env.NEXT_PUBLIC_VERCEL_URL
  ? `https://${process.env.NEXT_PUBLIC_VERCEL_URL}`
  : 'http://localhost:3000';

export const metadata = {
  metadataBase: new URL(defaultUrl),
  title: 'Better Reader',
  description: 'Helping you read better',
};

const geistSans = Geist({
  display: 'swap',
  subsets: ['latin'],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.className} h-screen overflow-hidden`} suppressHydrationWarning>
      <body className="bg-background text-foreground h-screen overflow-hidden">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <main className="h-full flex flex-col items-center overflow-hidden">
            <div className="flex-1 w-full flex flex-col gap-20 items-center overflow-hidden">
              <nav className="w-full flex justify-start border-b border-b-foreground/10 h-16">
                <div className="w-full max-w-5xl flex items-center p-3 px-5 text-sm">
                  <HeaderAuth />
                </div>
              </nav>

              {/* Children should now handle their own scrolling */}
              <div className="flex flex-col gap-20 max-w-5xl p-5 w-full h-full overflow-hidden">{children}</div>

              <footer className="w-full flex items-center justify-center border-t mx-auto text-center text-xs gap-8 py-16">
                <p>
                  Built with ❤️ by{' '}
                  <a href="https://github.com/mountaint0p" target="_blank" className="font-bold hover:underline">
                    Summit{' '}
                  </a>
                  and{' '}
                  <a href="https://github.com/ananyakgarg" target="_blank" className="font-bold hover:underline">
                    Ananya
                  </a>
                </p>
              </footer>
            </div>
          </main>
        </ThemeProvider>
      </body>
    </html>
  );
}
