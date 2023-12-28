#define DEVICE_NAME "Spider2"

#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4


#include <ArduinoBLE.h>
#include <PDM.h>
#include <SpiderThing_inferencing.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_HS300x.h>


/////////// FOR BLE COMMUNICATION ///////////////
BLEService ledService("19B10000-E8F2-537E-4F6C-111111111111"); // create service

// create switch characteristic ON
BLEByteCharacteristic switchCharacteristicON("19B10001-E8F2-537E-4F6C-111111111111", BLERead | BLEWrite);
// create switch characteristic OFF
BLEByteCharacteristic switchCharacteristicOFF("19B10001-E8F2-537E-4F6C-222222222222", BLERead | BLEWrite);

BLEStringCharacteristic responseCharacteristic("19B10002-E8F2-537E-4F6C-111111111111", BLERead | BLENotify, 200); // New characteristic for responses

/////////////////////////////////////////////////// CODE BELOW CAN BE DUPLICATED BETWEEN AGENTS ////////////////////////////
//////// FOR TEMPERATURE SENSOR ///////////

float old_temp = 0;
float old_hum = 0;


//////// FOR BAROMETRIC SENSOR ////////////
static double P0 = 101.325;


//////// FOR TEMPERATURE SENSOR ////////////
static double T0 = 1;

//////// FOR HUMIDITY SENSOR ////////////
static double H0 = 2;

///////// For Audio Event CLASSIFICATION ////////////
/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

float decibel_level = 0;
static int DECIBEL_LEVEL_THRESHOLD = 50;

// For Just Audio Level Sensor
// buffer to read samples into, each sample is 16-bits
short *sampleBufferAudioLevel;

// number of samples read
volatile int samplesRead;
volatile int avgSamplesRead;
volatile int new_category;
volatile int last_category;

const int ledPin = LED_BUILTIN; // pin to use for the LED

const int led1 = A0; 
const int led2 = A1; 
const int led3 = A2; 

static float value_prediction[EI_CLASSIFIER_LABEL_COUNT];

unsigned long previousMillis = 0;  // Stores the last time the action was performed
const long interval = 100;        // Interval at which to perform the action (milliseconds)


String floatToString(float value, int precision) {
    // Calculate buffer size (including the null terminator and potential '-' sign)
    int bufferSize = 10 + 1 + precision + 1 + 1;
    char buffer[bufferSize];

    // Convert the float to a string
    snprintf(buffer, sizeof(buffer), "%.*f", precision, value);

    // Return the result as a String object
    return String(buffer);
}

void setup() {
  // Serial.begin(9600);
  // while (!Serial);
  
  pinMode(ledPin, OUTPUT); // use the LED pin as an output
  pinMode(led1, OUTPUT); // use the LED pin as an output
  pinMode(led2, OUTPUT); // use the LED pin as an output
  pinMode(led3, OUTPUT); // use the LED pin as an output
  digitalWrite(led1, LOW);
  digitalWrite(led2, LOW);
  digitalWrite(led3, LOW);


  if (!HS300x.begin()) {
    // Serial.println("Failed to initialize humidity temperature sensor!");
    while (1);
  }

  if (!BARO.begin()) {
    // Serial.println("Failed to initialize pressure sensor!");
    while (1);
  }

  // begin initialization
  if (!BLE.begin()) {
    // Serial.println("starting Bluetooth® Low Energy module failed!");
    while (1);
  }

  // set the local name peripheral advertises
  BLE.setLocalName(DEVICE_NAME);
  // set the UUID for the service this peripheral advertises
  BLE.setAdvertisedService(ledService);

  // add the characteristic to the service
  ledService.addCharacteristic(switchCharacteristicON);

// add the characteristic to the service
  ledService.addCharacteristic(switchCharacteristicOFF);

  // Add the response characteristic to the service
  ledService.addCharacteristic(responseCharacteristic);

  // add service
  BLE.addService(ledService);

  // assign event handlers for connected, disconnected to peripheral
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);

  // assign event handlers for characteristic
  switchCharacteristicON.setEventHandler(BLEWritten, switchCharacteristicWrittenON);
  // set an initial value for the characteristic
  switchCharacteristicON.setValue(0);

  // assign event handlers for characteristic
  switchCharacteristicOFF.setEventHandler(BLEWritten, switchCharacteristicWrittenOFF);
  // set an initial value for the characteristic
  switchCharacteristicOFF.setValue(0);

  // start advertising
  BLE.advertise();

  // Serial.println(("Bluetooth® device active, waiting for connections..."));

  // START CLASSIFIER
  // summary of inferencing settings (from model_metadata.h)
  ei_printf("Inferencing settings:\n");
  ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                          sizeof(ei_classifier_inferencing_categories[0]));

  run_classifier_init();
  if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
      ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
      return;
  }

}

int avg_buffer = 0;


/////////// START INFERENCE HERE //////////////////
void process_audio_inference() {
  bool m = microphone_inference_record();
  if (!m) {
      // ei_printf("ERR: Failed to record audio...\n");
      return;
  }

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
  signal.get_data = &microphone_audio_signal_get_data;
  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
  if (r != EI_IMPULSE_OK) {
      ei_printf("ERR: Failed to run classifier (%d)\n", r);
      return;
  }

  if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {

      // print the predictions
      ei_printf("Predictions ");
      ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
          result.timing.dsp, result.timing.classification, result.timing.anomaly);
      ei_printf(": \n");

      for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
          value_prediction[ix] = result.classification[ix].value;
      }


#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

      print_results = 0;
  }
}

void process_barometric_monitor() {
  float pressure = BARO.readPressure();
  float altitude = 44330 * ( 1 - pow(pressure/P0, 1/5.255) );
  // Serial.println("Pressure: " + String(pressure));
  // Serial.println("Altitude: " + String(altitude));
}

void process_temperature_humidity_monitor() {

  // read all the sensor values
  float temperature = HS300x.readTemperature();
  float humidity    = HS300x.readHumidity();

  if (abs(old_temp - temperature) >= T0 || abs(old_hum - humidity) >= H0 )
  {
    String txt = "Temperature: " + String(temperature) + "\n";
    responseCharacteristic.writeValue(txt);
    txt = "Humidity: " + String(humidity) + "\n";
    responseCharacteristic.writeValue(txt);
  }

}

void loop() {
  // poll for Bluetooth® Low Energy events
  BLE.poll();
  process_audio_inference();
  // process_barometric_monitor();
  // process_temperature_humidity_monitor();
  process_publish();
}

void process_publish() {
    // Calculate decibel level
    unsigned long currentMillis = millis();
    // if (currentMillis - previousMillis >= interval) {
          // save the last time you performed the action
          // previousMillis = currentMillis;
          if(decibel_level > DECIBEL_LEVEL_THRESHOLD) {
            char str[50]; // Buffer large enough to hold the float string
            snprintf(str, sizeof(str), "{'event': 'decibel', 'decibel': '%f'}\n", decibel_level);

            // Now 'str' is a 'const char*' pointing to the string representation of the float
            const char* floatString = str;
            ei_printf(str);
            responseCharacteristic.writeValue(floatString);  // Send "OFF" response

            String txt = "{'event': 'prediction'";
            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                txt += ", '" + String(ei_classifier_inferencing_categories[ix]) + "': '"+ String(value_prediction[ix]) + "'";
            }
            txt += "}\n";
            const char* cstr = txt.c_str();
            ei_printf(cstr);
            responseCharacteristic.writeValue(txt);

          }
    // }
}

void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  // Serial.print("Connected event, central: ");
  // Serial.println(central.address());
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  // Serial.print("Disconnected event, central: ");
  // Serial.println(central.address());
}

void switchCharacteristicWrittenON(BLEDevice central, BLECharacteristic characteristic) {
  // central wrote new value to characteristic, update LED
  // Serial.print("Characteristic event, recv: ");

  char str[50]; // Buffer large enough to hold the float string
  int led_num = 0;
  int c = 1;
  if (switchCharacteristicON.value() == 0x01) {
    // Serial.println("1");
    digitalWrite(led1, LOW);
    led_num = 1;
  } 

  if (switchCharacteristicON.value() == 0x02) {
    // Serial.println("1");
    digitalWrite(led2, LOW);
    led_num = 2;
  } 

  if (switchCharacteristicON.value() == 0x03) {
    // Serial.println("1");
    digitalWrite(led3, LOW);
    led_num = 3;
  } 
  snprintf(str, sizeof(str), "{'event': 'led', 'led': '%d', 'status': '%d'}\n", led_num, c);
  const char* sendString = str;
  responseCharacteristic.writeValue(sendString);  // Send "OFF" response
}

void switchCharacteristicWrittenOFF(BLEDevice central, BLECharacteristic characteristic) {
  // central wrote new value to characteristic, update LED
  // Serial.print("Characteristic event, recv: ");

  char str[50]; // Buffer large enough to hold the float string
  int led_num = 0;
  int c = 0;
  if (switchCharacteristicOFF.value() == 0x01) {
    // Serial.println("1");
    digitalWrite(led1, HIGH);
    led_num = 1;
  } 

  if (switchCharacteristicOFF.value() == 0x02) {
    // Serial.println("1");
    digitalWrite(led2, HIGH);
    led_num = 2;
  } 

  if (switchCharacteristicOFF.value() == 0x03) {
    // Serial.println("1");
    digitalWrite(led3, HIGH);
    led_num = 3;
  } 

  snprintf(str, sizeof(str), "{'event': 'led', 'led': '%d', 'status': '%d'}\n", led_num, c);
  const char* sendString = str;
  responseCharacteristic.writeValue(sendString);  // Send "OFF" response

  // responseCharacteristic.writeValue(ret);  // Send "OFF" response
}

//////////// Edge Impulse Generated ///////////////////////

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);
    // PDM.onReceive(onPDMdata);
    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));


    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    // PDM.setGain(127);
    PDM.setGain(255);

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        // ei_printf(
        //     "Error sample buffer overrun. Decrease the number of slices per model window "
        //     "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

float calculate_decibel_level(signed short *samples, size_t length) {
    float rms = 0.0;
    for (size_t i = 0; i < length; i++) {
        rms += pow(samples[i], 2);
    }
    rms = sqrt(rms / length);

    // Assuming 1 as reference value for RMS. This might need to be adjusted based on your specific use case
    float referenceRMS = 1.0;
    float decibel = 20 * log10(rms / referenceRMS); 

    return decibel;
}


/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= interval) {
      previousMillis = currentMillis;
      decibel_level = calculate_decibel_level(&inference.buffers[inference.buf_select ^ 1][offset], length);
    }
    
    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}


/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }

    }

}


#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
