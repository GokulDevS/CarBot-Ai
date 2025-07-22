import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chatbot.html',
  styleUrls: ['./chatbot.css']
})
export class ChatbotComponent {
  messages: { sender: string, text: string }[] = [
    { sender: 'bot', text: 'Hi! How can I help you today?' }
  ];
  userInput: string = '';

  constructor(private http: HttpClient) {}

  sendMessage() {
    const message = this.userInput.trim();
    if (!message) return;

    this.messages.push({ sender: 'user', text: message });

    
    this.userInput = '';

    
    this.messages.push({ sender: 'bot', text: 'Typing...' });

    
    this.http.post<any>('http://127.0.0.1:8000/predict-intent', { message }).subscribe(
      (response) => {
        
        this.messages.pop();
        this.messages.push({ sender: 'bot', text: response.response });
      },
      (error) => {
        this.messages.pop();
        this.messages.push({ sender: 'bot', text: "Oops! Something went wrong." });
        console.error(error);
      }
    );
  }
}
