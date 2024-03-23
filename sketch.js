let imgColor, imgGray;
let fontInterRegular, fontInterMedium, fontInterSemiBold;
let gfx; // Declare the off-screen graphics buffer
let poseNet;
let brain;
let currentDetectedPose, targetPosesSet;
let poseMatched = false;
let thumbsUpImage;
let timeRevealed = false;
let video;

const canvasWidth = 1620;
const canvasHeight = 880;
const clockFrameWidth = 820;
const clockFrameHeight = 760;
const cameraFrameWidth = 600;
const cameraFrameHeight = 450;
const imgWidth = 600;
const imgHeight = 450;
const imageX = (clockFrameWidth - imgWidth) / 2;
const clockSliderX = 110;
const clockSliderY = 110;
const borderRadius = 10;
const interactionFrameWidth = 800;
const interactionFrameHeight = 880;
const videoWidth = 800;
const videoHeight = 600;
const borderRadiusInteractionControls = 40;
const poseImageWidth = 140;
const poseImageHeight = 140;
const poseImageFrameBorderRadius = 20;
const defaultPoseRectColor = "#ededed";
const performedPoseRectColor = "#abf7b4";
const currentPoseRectColor = "#ccefff";

// Define university opening hours for each day of the week
// Array starts with Monday; each object has open and close times (24-hour format)
const uniHours = [
  { open: 8, close: 23 }, // Monday
  { open: 8, close: 23 }, // Tuesday
  { open: 8, close: 23 }, // Wednesday
  { open: 8, close: 23 }, // Thursday
  { open: 8, close: 23 }, // Friday
  { open: 0, close: 0 },  // Saturday (closed)
  { open: 0, close: 0 }   // Sunday (closed)
];
// if you want to set the time that is not a round number, you can use the stringTimeToDecimal function
// e.g. { open: 8, close: stringTimeToDecimal('23:30') } // Tuesday

// all the poses that the model is trained on
const poses = [
  'Hands & legs separated',
  'Half squat - upper body flex',
  'Stretch Hands straight up',
  'Candle Pose',
  'Huddle Pose',
  'Left-leg & right-arm triangles',
  'Halo around your head',
  'Stretch back (right side)',
  'Stretch back (left side)'
];

const poseImages = [];
let targetPoseImages;

// preloading all the assets used in the project
function preload() {
  imgColor = loadImage('./images/buildings of thuas - colored.png');
  imgGray = loadImage('./images/buildings of thuas - grayscale.png');
  thumbsUpImage = loadImage('./images/thumbs-up icon.png');
  fontInterRegular = loadFont('./fonts/inter/static/Inter-Regular.ttf');
  fontInterMedium = loadFont('./fonts/inter/static/Inter-Medium.ttf');
  fontInterSemiBold = loadFont('./fonts/inter/static/Inter-SemiBold.ttf');
  loadPoseImages();
}

// loading the images of the poses
function loadPoseImages() {
  for (let i = 0; i < poses.length; i++) {
    poseImages.push(loadImage(`./images/${poses[i]}.png`));
  }
}

function setup() {
  createCanvas(canvasWidth, canvasHeight); // Set the canvas size to match your image size

  video = createCapture(VIDEO);
  video.size(videoWidth, videoHeight);
  video.hide();

  // the use of arrow function here(nothing than the shorter way of writing a function)
  poseNet = ml5.poseNet(video, () => console.log('PoseNet loaded!'));
  poseNet.on('pose', getPoses);
  
  // defining the options for the model to be used for the classification
  const options = {
    inputs: 34, // 17 keypoints * 2 (x and y)
    outputs: 9, // 9 poses
    task: "classification",
    debug: true,
  };
  
  brain = ml5.neuralNetwork(options);

  // custom model info
  const modelInfo = {
    model: "./model/model.json",
    metadata: "./model/model_meta.json",
    weights: "./model/model.weights.bin",
  };

  brain.load(modelInfo, CustomModelReady);

  // Create an off-screen graphics buffer to draw the color image onto
  gfx = createGraphics(imgWidth + clockSliderX, imgHeight + clockSliderY);
  gfx.image(imgColor, clockSliderX, clockSliderY); // Draw the color image onto the off-screen graphics
}

function CustomModelReady() {
  console.log("Custom Model is ready!");
}

function classifyPose(pose) {
  // preparing the inputs for the model to classify the pose
  let inputs = [];
  for (let i = 0; i < pose.keypoints.length; i++) {
    // this is the same as the inputs for the model when it was trained
    const x = pose.keypoints[i].position.x;
    const y = pose.keypoints[i].position.y;
    inputs.push(x);
    inputs.push(y);
  }
  brain.classify(inputs, gotResults);
}

function gotResults(error, results) {
  // when the pose is classified(compared to one of the poses in the set of poses the model is trained on)...
  
  // if there is an error, log it to the console
  if (error) {
    console.error(error);
    return;
  }

  // otherwise, if the model is 98% sure that the pose is one of the poses in the set of poses the model is trained on, 
  // set that pose as the detected pose. Instead, if the model is not sure enough, set the detected pose to an empty string
  // you can adjust this value to be more or less strict(for me, this value seemed perfect)
  if (results[0].confidence > 0.98) {
    currentDetectedPose = results[0].label;
  } else {
    currentDetectedPose = "";
  }
}

function getPoses(poses) {
  // when the pose/poses is/are detected, classify each of the poses
  if (poses.length > 0) {
    poses.forEach(p => {
      classifyPose(p.pose);
    });
  }
}

function draw() {
  background("#ffffff"); // Light blue background

  // CLOCK PART(LEFT PART)

  // Draw the grayscale image
  image(imgGray, clockSliderX, clockSliderY, imgWidth, imgHeight);

  // Get current day of the week (0 for Sunday, 6 for Saturday)
  let currentDayOfWeek = (new Date()).getDay(); // Gets the current day of the week (0-6)
  let dayOfWeek = (currentDayOfWeek + 6) % 7; // Adjusting so Monday is 0, Sunday is 6

  // Get opening and closing hours for the current day
  let { open, close } = uniHours[dayOfWeek];
  
  // Calculate the current time and map it to the range of opening hours
  let now = hour() + minute() / 60 + second() / 3600;
  let dayDuration = close - open;
  let timeSinceOpen = now - open;

  // Calculate the fraction of the day passed within university opening hours
  let dayFraction = dayDuration > 0 ? constrain(timeSinceOpen / dayDuration, 0, 1) : 0;

  // Map the dayFraction to the slider's(imgWidth) width, not the entire canvas
  // and offset by clockSliderX to position correctly within the canvas
  let splitX = clockSliderX + (dayFraction * imgWidth);
  
  // Ensure the reveal effect stays within the image boundaries
  splitX = constrain(splitX, clockSliderX, clockSliderX + imgWidth);

  // Use the calculated splitX to draw a portion of the gfx
  image(gfx, clockSliderX, clockSliderY, splitX - clockSliderX, imgHeight, clockSliderX, clockSliderY, splitX - clockSliderX, imgHeight);

  stroke(0);
  strokeWeight(2);
  line(splitX, clockSliderY, splitX, imgHeight + clockSliderY); // Draw line at the split position between color and grayscale images

  // printMainTitle("time");
  // printMainTitle();
  if (timeRevealed) {
    // show the time as the main title of the clock for 10 seconds if all three poses are matched
    printMainTitle("time");
  } else {
    // show the default title of the clock otherwise
    printMainTitle();
  }
  printLeftClockSliderLine(decimalTimeToString(open));
  printRightClockSliderLine(decimalTimeToString(close));
  printLeftSignifier(splitX);
  printRightSignifier(splitX);
  printTodaysDate();

  // write instructions for the interaction part
  const instructionsY = clockSliderY + imgHeight + 60;
  writeCenteredTextUsingVariousFontWeights("I want to know the exact time... ", "Let’s stretch a bit :)", instructionsY, 24);
  writeCenteredTextUsingVariousFontWeights("Take a look at ", "the TV screen just ahead of you.", instructionsY + 55, 20);

  printTitleForClockSlider();

  // INTERACTION PART(RIGHT PART)
  applyMoreNaturalVideoFeel();
  image(video, -clockFrameWidth, 280, videoWidth, videoHeight);
  noStroke();

  // if the target poses are not set yet, set them
  if (!targetPosesSet) {
    setTargetPoses();
    targetPosesSet = true;
  }

  // render the interaction part of the screen
  if (targetPosesSet) {
    printInteractionInstructions("Do these ", "three poses ", "to get the ", "exact time.");
    renderPoseImagesBg();
    renderPoseImages(...targetPoseImages);
    // render the current pose image across the camera frame if the time is not showing at the moment
    !timeRevealed && renderCurrentPoseImageAcrossTheCameraFrame();
  }

  // leaving a green background over the camera frame for a while if the pose is matched
  if (poseMatched) {
    fill(performedPoseRectColor);
    noStroke();
    rect(clockFrameWidth, 0, interactionFrameWidth, interactionFrameHeight);
  }

  // if person is in the pose that the program is currently checking on...
  const currentSetPose = targetPoseImages.find(pose => pose.state == "current")?.poseName;
  if (currentSetPose == currentDetectedPose && !timeRevealed) {
    poseMatched = true;

    // show a green background over the camera frame
    fill(performedPoseRectColor);
    noStroke();
    rect(clockFrameWidth, 280, interactionFrameWidth, interactionFrameHeight);

    const lastPoseMatched = targetPoseImages[2].state == "current";
    if (lastPoseMatched) {
      // if the last pose is matched, render the time for 10 seconds
      timeRevealed = true;
      targetPoseImages[2].state = "performed";

      // and set the new set of poses after 10 seconds
      setTimeout(() => {
        setTargetPoses();
        timeRevealed = false;
      }, 10000);
    } else {
      // otherwise, set the next pose as the current pose
      const currentTargetPoseIndex = targetPoseImages.findIndex(pose => pose.state == "current");
      targetPoseImages[currentTargetPoseIndex].state = "performed";
      targetPoseImages[currentTargetPoseIndex + 1].state = "current";
    }

    // leaving a green background for a while
    setTimeout(() => {
      poseMatched = false;
    }, 300);
  }

  // add separator line in the middle of the two screens
  stroke("#000000");
  strokeWeight(2);
  line(clockFrameWidth, 0, clockFrameWidth, canvasHeight);

  // add limiter lines on the bottom edge of the clock frame and the right edge of the interaction frame
  stroke("#000000");
  strokeWeight(2);
  line(0, clockFrameHeight, clockFrameWidth, clockFrameHeight);
  line(canvasWidth - 1, 0, canvasWidth - 1, canvasHeight);
}

function renderCurrentPoseImageAcrossTheCameraFrame() {
  const currentPoseImage = targetPoseImages.find(pose => pose.state == "current")?.image;
  // rectangle behind the image
  fill(currentPoseRectColor);
  noStroke();
  tint(255, 127); // Apply transparency to the image
  image(currentPoseImage, clockFrameWidth + 100, 280, videoHeight, videoHeight);
  noTint(); // Remove transparency so it doesn't affect other images
}

function setTargetPoses() {
  // getting 3 unique random indexes for the poses
  const poseIndexes = getUniqueRandomNumbers(3, 0, poses.length - 1);
  // setting the target poses to be displayed on the screen
  targetPoseImages = [
    { image: poseImages[poseIndexes[0]], poseName: poses[poseIndexes[0]], state: "current" },
    { image: poseImages[poseIndexes[1]], poseName: poses[poseIndexes[1]], state: "default" },
    { image: poseImages[poseIndexes[2]], poseName: poses[poseIndexes[2]], state: "default" }
  ];
}

function getUniqueRandomNumbers(numberOfNumbers, min, max) {
  let numbers = new Set();
  while (numbers.size < numberOfNumbers) {
    numbers.add(Math.floor(Math.random() * (max - min + 1)) + min);
  }
  return [...numbers];
}

function printInteractionInstructions(textRegular1, textBold1, textRegular2, textBold2) {
  // getting back to the normal way of painting the image to the screen
  translate(interactionFrameWidth, 0);
  scale(-1, 1);

  // white rectangle behind the text
  fill("#ffffff");
  noStroke();
  rect(clockFrameWidth, 0, interactionFrameWidth, 80);

  // render the text
  fill("#000000");
  noStroke();
  textAlign(LEFT, TOP);
  textSize(24);

  // Calculate the width of each part of the text with appropriate style
  textFont(fontInterRegular);
  const textRegular1Width = textWidth(textRegular1);
  textFont(fontInterSemiBold);
  const textBold1Width = textWidth(textBold1);
  textFont(fontInterRegular);
  const textRegular2Width = textWidth(textRegular2);
  textFont(fontInterSemiBold);
  const textBold2Width = textWidth(textBold2);

  // Calculate the total width of the text
  const totalWidth = textRegular1Width + textBold1Width + textRegular2Width + textBold2Width;

  // Calculate the starting x position to center the text
  let x = clockFrameWidth + (interactionFrameWidth - totalWidth) / 2;
  const y = 40;

  // Draw the texts with appropriate styles
  textFont(fontInterRegular);
  text(textRegular1, x, y);
  x += textRegular1Width;

  textFont(fontInterSemiBold);
  text(textBold1, x, y);
  x += textBold1Width;

  textFont(fontInterRegular);
  text(textRegular2, x, y);
  x += textRegular2Width;

  textFont(fontInterSemiBold);
  text(textBold2, x, y);
}

function renderPoseImagesBg() {
  // rectangle behind the images
  fill("#ffffff");
  noStroke();
  rect(clockFrameWidth, 80, interactionFrameWidth, 200);
}

function renderPoseImages(poseImageLeftInfo, poseImageCenterInfo, poseImageRightInfo) {
  const poseImagesGap = 35;

  // render left image
  renderPoseImage(poseImageLeftInfo.image, clockFrameWidth + (interactionFrameWidth - poseImageWidth) / 2 - poseImageWidth - poseImagesGap, 100, poseImageLeftInfo.state);

  // render center image
  renderPoseImage(poseImageCenterInfo.image, clockFrameWidth + (interactionFrameWidth - poseImageWidth) / 2, 100, poseImageCenterInfo.state);

  // render right image
  renderPoseImage(poseImageRightInfo.image, clockFrameWidth + (interactionFrameWidth - poseImageWidth) / 2 + poseImageWidth + poseImagesGap, 100, poseImageRightInfo.state);
}

function renderPoseImage(poseImage, x, y, state) {
  // rectangle behind the image
  // color of the rectangle depends on the state of the pose
  if (state == "performed") {
    fill(performedPoseRectColor);
  } else if (state == "default") {
    fill(defaultPoseRectColor);
  } else {
    fill(currentPoseRectColor);
  }
  noStroke();
  rect(x, y, poseImageWidth, poseImageHeight, poseImageFrameBorderRadius);

  // render image
  image(poseImage, x, y, poseImageWidth, poseImageHeight);

  // render the check mark
  if (state == "performed") {
    image(thumbsUpImage, x + poseImageWidth - 35, y + poseImageHeight - 35, 50, 50);
  }
}

// partially pasted from ChatGPT4 example
// flipping the screen according to the y axis (more natural feel)
function applyMoreNaturalVideoFeel() {
  // starting to paint the image from the right side of the screen
  translate(800, 0);
  // setting the direction of painting the image to the screen to right-to-left
  scale(-1, 1);
}

function printTitleForClockSlider() {
  // draw the rectangle for the clock slider
  fill(255);
  noStroke();
  rect(clockSliderX + 190, clockSliderY, 218, 40, 0, 0, borderRadius, borderRadius);
  // print the title
  fill(0);
  textFont(fontInterRegular);
  textSize(18);
  textAlign(CENTER, CENTER);
  text('THUAS Opening Hours', clockFrameWidth / 2, clockSliderY + 18);
}

// writes the text in one row in the center of the canvas using various font weights
function writeCenteredTextUsingVariousFontWeights(textRegular, textBold, y, fontSize, targetFrameWidth = clockFrameWidth) {
  fill(0);
  noStroke();

  // Calculate width of each part of the text with appropriate style
  textFont(fontInterRegular);
  textSize(fontSize);
  let textRegularWidth = textWidth(textRegular);
  textFont(fontInterSemiBold);
  textSize(fontSize);
  let textBoldWidth = textWidth(textBold);

  // Total width of the text
  let totalWidth = textRegularWidth + textBoldWidth;

  // Calculate starting x position to center the text
  let x = (targetFrameWidth - totalWidth) / 2;

  // Draw the first part of the text
  textFont(fontInterRegular);
  textSize(fontSize);
  text(textRegular, x, y);
  x += textRegularWidth;

  // Draw the bold part of the text
  textFont(fontInterSemiBold);
  textSize(fontSize);
  text(textBold, x, y);
}

function printLeftSignifier(splitX) {
  // rectangle for the signifier
  fill("#ffffff");
  noStroke();
  const rectWidth = 90;
  const rectHeight = 115;
  rect(splitX - rectWidth - 1, clockSliderY + 100, rectWidth, rectHeight, borderRadius, 0, 0, borderRadius);
  // print the signifier text
  fill(0);
  textFont(fontInterRegular);
  textSize(18);
  textAlign(RIGHT, TOP);
  text("Time\nSince\nOpen", splitX - rectWidth / 2 + 30, clockSliderY + 116);
  // arrow pointing to the left
  stroke("#000000");
  strokeWeight(1);
  const clockSliderYAddition = 195;
  line(splitX - 10, clockSliderY + clockSliderYAddition, splitX - 70, clockSliderY + clockSliderYAddition);
  line(splitX - 70, clockSliderY + clockSliderYAddition, splitX - 60, clockSliderY + clockSliderYAddition - 5);
  line(splitX - 70, clockSliderY + clockSliderYAddition, splitX - 60, clockSliderY + clockSliderYAddition + 5);
}

function printRightSignifier(splitX) {
  // rectangle for the signifier
  fill("#ffffff");
  noStroke();
  const rectWidth = 90;
  const rectHeight = 115;
  rect(splitX + 1, clockSliderY + 100, rectWidth, rectHeight, 0, borderRadius, borderRadius, 0);
  // print the signifier text
  fill(0);
  textFont(fontInterRegular);
  textSize(18);
  textAlign(LEFT, TOP);
  text("Time\nUntil\nClose", splitX + rectWidth / 2 - 30, clockSliderY + 116);
  // arrow pointing to the right
  stroke("#000000");
  strokeWeight(1);
  const clockSliderYAddition = 195;
  line(splitX + 10, clockSliderY + clockSliderYAddition, splitX + 70, clockSliderY + clockSliderYAddition);
  line(splitX + 70, clockSliderY + clockSliderYAddition, splitX + 60, clockSliderY + clockSliderYAddition - 5);
  line(splitX + 70, clockSliderY + clockSliderYAddition, splitX + 60, clockSliderY + clockSliderYAddition + 5);
}


function printMainTitle(type = "text") {
  if (type === "text") {
    fill(0);
    noStroke();
    textFont(fontInterRegular);
    textSize(32);
    textAlign(CENTER, CENTER);
    text('Time of the Day', clockFrameWidth / 2, 55);
  } else {
    printTime();
  }
}

function printLeftClockSliderLine(openingTime) {
  // draw the rectangle behind the text
  fill(255);
  noStroke();
  const rectWidth = 65;
  const rectHeight = 26;
  rect(clockSliderX, clockSliderY - 1, rectWidth, rectHeight, 0, 0, borderRadius, 0);
  // print the text
  fill(0);
  textFont(fontInterMedium);
  textSize(18);
  text(openingTime, clockSliderX + 32, clockSliderY + 8);
  // draw the line/limiter
  stroke(0);
  strokeWeight(2);
  line(clockSliderX, clockSliderY, clockSliderX, imgHeight + clockSliderY);
}

function printRightClockSliderLine(closingTime) {
  // draw the rectangle behind the text
  fill(255);
  noStroke();
  const rectWidth = 65;
  const rectHeight = 26;
  rect(clockSliderX + imgWidth - rectWidth, clockSliderY - 1, rectWidth, rectHeight, 0, 0, 0,  borderRadius);
  // print the text
  fill(0);
  textFont(fontInterMedium);
  textSize(18);
  noStroke();
  text(closingTime, clockSliderX + imgWidth - 32, clockSliderY + 8);
  // draw the line/limiter
  stroke(0);
  strokeWeight(2);
  line(clockSliderX + imgWidth, clockSliderY, clockSliderX + imgWidth, imgHeight + clockSliderY);
}

function printTodaysDate() {
  // format: February 12, 2024
  let date = new Date();
  let month = date.toLocaleString('default', { month: 'long' });
  let day = date.getDate();
  let year = date.getFullYear();
  let formattedDate = month + ' ' + day + ', ' + year;
  // rectangle for the date
  fill(255);
  noStroke();
  const rectWidth = 240;
  const rectHeight = 50;
  rect(clockSliderX + (imgWidth - rectWidth) / 2, clockSliderY + imgHeight - rectHeight + 1, rectWidth, rectHeight, borderRadius, borderRadius, 0, 0);
  // print the date
  fill(0);
  noStroke();
  textFont(fontInterMedium);
  textSize(20);
  // getting the width of the text to center it
  let dateTextWidth = textWidth(formattedDate);
  textAlign(LEFT, TOP);
  text(formattedDate, clockSliderX + (imgWidth - dateTextWidth) / 2, clockSliderY + imgHeight - rectHeight + 14);
}

function printTime() {
  // Get the current time
  let h = hour();
  let m = minute();
  let s = second();
  if (h < 10) h = '0' + h;
  if (m < 10) m = '0' + m;
  if (s < 10) s = '0' + s;
  let time = h + ':' + m + ':' + s;

  // Draw the time
  fill(0);
  noStroke();
  textFont(fontInterRegular);
  textSize(32);
  textAlign(CENTER, CENTER);
  text(time, clockFrameWidth / 2, 46);
}

// entirely pasted from ChatGPT4 example
function decimalTimeToString(decimalTime) {
  // Extract the hour part by flooring the decimal time
  const hours = Math.floor(decimalTime);
  
  // Extract the minutes part by subtracting the hours from the decimal time,
  // then multiplying by 60 to convert to minutes
  const minutes = Math.round((decimalTime - hours) * 60);
  
  // Format the hours and minutes with leading zeros if needed
  const formattedHours = hours.toString().padStart(2, '0');
  const formattedMinutes = minutes.toString().padStart(2, '0');
  
  // Combine the formatted hours and minutes into a HH:MM string
  return `${formattedHours}:${formattedMinutes}`;
}

// entirely pasted from ChatGPT4 example
function stringTimeToDecimal(timeString) {
  // Split the time string into hours and minutes
  const [hours, minutes] = timeString.split(':').map(Number);
  
  // Convert hours and minutes into decimal
  const decimalTime = hours + (minutes / 60);
  
  // Return the decimal time
  return decimalTime;
}