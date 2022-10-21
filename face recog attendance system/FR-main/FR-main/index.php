<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Reco</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <style>
        * {
          box-sizing: border-box;
        }
        
        body {
          margin: 0px;
          font-family: 'segoe ui';
        }
        
        .nav {
          height: 50px;
          width: 100%;
          background-color: #4d4d4d;
          position: relative;
        }
        
        .nav > .nav-header {
          display: inline;
        }
        
        .nav > .nav-header > .nav-title {
          display: inline-block;
          font-size: 22px;
          color: #fff;
          padding: 10px 10px 10px 10px;
        }
        
        .nav > .nav-btn {
          display: none;
        }
        
        .nav > .nav-links {
          display: inline;
          float: right;
          font-size: 18px;
        }
        
        .nav > .nav-links > a {
          display: inline-block;
          padding: 13px 10px 13px 10px;
          text-decoration: none;
          color: #efefef;
        }
        
        .nav > .nav-links > a:hover {
          background-color: rgba(0, 0, 0, 0.3);
        }
        
        .nav > #nav-check {
          display: none;
        }
        
        @media (max-width:600px) {
          .nav > .nav-btn {
            display: inline-block;
            position: absolute;
            right: 0px;
            top: 0px;
          }
          .nav > .nav-btn > label {
            display: inline-block;
            width: 50px;
            height: 50px;
            padding: 13px;
          }
          .nav > .nav-btn > label:hover,.nav  #nav-check:checked ~ .nav-btn > label {
            background-color: rgba(0, 0, 0, 0.3);
          }
          .nav > .nav-btn > label > span {
            display: block;
            width: 25px;
            height: 10px;
            border-top: 2px solid #eee;
          }
          .nav > .nav-links {
            position: absolute;
            display: block;
            width: 100%;
            background-color: #333;
            height: 0px;
            transition: all 0.3s ease-in;
            overflow-y: hidden;
            top: 50px;
            left: 0px;
          }
          .nav > .nav-links > a {
            display: block;
            width: 100%;
          }
          .nav > #nav-check:not(:checked) ~ .nav-links {
            height: 0px;
          }
          .nav > #nav-check:checked ~ .nav-links {
            height: calc(100vh - 50px);
            overflow-y: auto;
          }
        }


        
            </style>
        </head>
        <body>
          <div class="nav">
            <input type="checkbox" id="nav-check">
            <div class="nav-header">
              <div class="nav-title">
                Face Recognition App
              </div>
            </div>
            <div class="nav-btn">
              <label for="nav-check">
                <span></span>
                <span></span>
                <span></span>
              </label>
            </div>
            
            <div class="nav-links">
                
              <a href="signin.php" >Admin</a>
                    </div>
          </div>
    <div class="h1-class">
        
          <h1>Face Recognition App</h1>
    </div>
    <div id="imglogo">
    <a href="panalselector.php"><img src="images/logo.svg" width="100%" height="100%" class="logoimg">
        </a>
        </div>

   
</body>
</html>