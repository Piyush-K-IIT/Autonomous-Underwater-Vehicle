#include <Servo.h>
Servo ESC;

// Global variables
float value = 0;    

// For 270° potentiometer
const int MIN_ADC = 0;
const int MAX_ADC = 1024;  // STM32 has 12-bit ADC (0-4095)
const int CENTER = (MIN_ADC + MAX_ADC) / 2;
const int DEAD_ZONE = 100;  // Increased dead zone for more stability

// ESC PWM values (in microseconds)
const int ESC_MIN = 1000;   // Minimum PWM value
const int ESC_MAX = 2000;   // Maximum PWM value
const int ESC_NEUTRAL = 1500; // Neutral position

void setup() {
  Serial.begin(115200);  // Higher baud rate for better performance
  
  // Configure ESC on PA0
  ESC.attach(PA0, ESC_MIN, ESC_MAX);  // Explicitly set PWM range
  
  // Configure ADC input
  pinMode(PA1, INPUT_ANALOG);
  
  // Initialize ESC to neutral position
  ESC.writeMicroseconds(ESC_NEUTRAL);
  delay(1000);  // Give ESC time to initialize
}

void loop() { 
  // Read potentiometer value multiple times for stability
  value = 0;
  for(int i = 0; i < 20; i++) {  // Reduced samples for faster response
    value += analogRead(PA1);
    delay(1);
  }  
  value = value / 20;  // Calculate average
  
  // Constrain the value to our expected range
  value = constrain(value, MIN_ADC, MAX_ADC);
  
  // Create a dead zone in the center
  if (value >= CENTER - DEAD_ZONE && value <= CENTER + DEAD_ZONE) {
    // Neutral zone - stop motor
    ESC.writeMicroseconds(ESC_NEUTRAL);
    Serial.println("Neutral: 1500us");
  }
  else if (value < CENTER - DEAD_ZONE) {
    // Reverse direction 
    int pwmValue = map(value, MIN_ADC, CENTER - DEAD_ZONE, ESC_MIN, ESC_NEUTRAL);
    ESC.writeMicroseconds(pwmValue);
    
    Serial.print("Reverse PWM: ");
    Serial.println(pwmValue);
  }
  else {
    // Forward direction
    int pwmValue = map(value, CENTER + DEAD_ZONE, MAX_ADC, ESC_NEUTRAL, ESC_MAX);
    ESC.writeMicroseconds(pwmValue);
    
    Serial.print("Forward PWM: ");
    Serial.println(pwmValue);
  }
  
  delay(10);  // Small delay for stability
}
