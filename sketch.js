let poseNet;
let brain;
let currentDetectedPose, targetPosesSet;
let poseMatched = false;
let fontInterRegular, fontInterSemiBold;
let thumbsUpImage;
let timeRevealed = false;

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

function preload() {
  thumbsUpImage = loadImage('./images/thumbs-up icon.png');
  fontInterRegular = loadFont('./fonts/inter/static/Inter-Regular.ttf');
  fontInterSemiBold = loadFont('./fonts/inter/static/Inter-SemiBold.ttf');
}

function setup() {
  createCanvas(interactionFrameWidth, interactionFrameHeight);
  
  loadPoseImages();

  // targetPoseImages = [
  //   { image: poseImages[0], state: "current" },
  //   { image: poseImages[1], state: "default" },
  //   { image: poseImages[2], state: "default" }
  // ];

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
}

// loading the images of the poses
function loadPoseImages() {
  for (let i = 0; i < poses.length; i++) {
    poseImages.push(loadImage(`./images/${poses[i]}.png`));
  }
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
  applyMoreNaturalVideoFeel();
  background("#c4ecff");
  image(video, 0, 280, videoWidth, videoHeight);
  noStroke();

  if (timeRevealed) {
    // console.log current time
    const time = new Date();
    const hours = time.getHours();
    const minutes = time.getMinutes();
    const seconds = time.getSeconds();
    const timeString = `${hours}:${minutes}:${seconds}`;
    console.log(timeString);
  }

  if (!targetPosesSet) {
    setTargetPoses();
    targetPosesSet = true;
  }

  if (targetPosesSet) {
    printInteractionInstructions("Do these ", "three poses ", "to get the ", "exact time.");
    renderPoseImagesBg();
    renderPoseImages(...targetPoseImages);
  }

  // leaving a green background for a while if the pose is matched
  if (poseMatched) {
    background(performedPoseRectColor);
  }

  // if person is in the currently checking pose...
  const currentSetPose = targetPoseImages.find(pose => pose.state == "current")?.poseName;
  if (currentSetPose == currentDetectedPose && !timeRevealed) {
    poseMatched = true;
    background(performedPoseRectColor);

    const lastPoseMatched = targetPoseImages[2].state == "current";
    if (lastPoseMatched) {
      // if the last pose is matched, set a new set of poses
      timeRevealed = true;
      targetPoseImages[2].state = "performed";

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
}

function setTargetPoses() {
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
  // rect(0, 0, interactionFrameWidth, 100, 0, 0, borderRadiusInteractionControls, borderRadiusInteractionControls);
  rect(0, 0, interactionFrameWidth, 80);

  // text
  fill("#000000");
  noStroke();
  textAlign(LEFT, TOP);
  textSize(24);

  textFont(fontInterRegular);
  const textRegular1Width = textWidth(textRegular1);
  textFont(fontInterSemiBold);
  const textBold1Width = textWidth(textBold1);
  textFont(fontInterRegular);
  const textRegular2Width = textWidth(textRegular2);
  textFont(fontInterSemiBold);
  const textBold2Width = textWidth(textBold2);

  const totalWidth = textRegular1Width + textBold1Width + textRegular2Width + textBold2Width;

  let x = (interactionFrameWidth - totalWidth) / 2;
  const y = 40;

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
  rect(0, 80, interactionFrameWidth, 200);
}

function renderPoseImages(poseImageLeftInfo, poseImageCenterInfo, poseImageRightInfo) {
  const poseImagesGap = 35;

  // render left image
  renderPoseImage(poseImageLeftInfo.image, (interactionFrameWidth - poseImageWidth) / 2 - poseImageWidth - poseImagesGap, 100, poseImageLeftInfo.state);

  // render center image
  renderPoseImage(poseImageCenterInfo.image, (interactionFrameWidth - poseImageWidth) / 2, 100, poseImageCenterInfo.state);

  // render right image
  renderPoseImage(poseImageRightInfo.image, (interactionFrameWidth - poseImageWidth) / 2 + poseImageWidth + poseImagesGap, 100, poseImageRightInfo.state);
}

function renderPoseImage(poseImage, x, y, state) {
  // rectangle behind the image
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

// flipping the screen according to the y axis (more natural feel)
function applyMoreNaturalVideoFeel() {
  // starting to paint the image from the right side of the screen
  translate(800, 0);
  // setting the direction of painting the image to the screen to right-to-left
  scale(-1, 1);
}