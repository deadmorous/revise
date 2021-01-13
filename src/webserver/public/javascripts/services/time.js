/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

(function(){
    class AnimationPlayer extends EventTarget {
        constructor() {
            super();

            this.playPauseBtnJQ = $("#play-pause-btn");
            this.previousBtnJQ  = $("#previous-btn");
            this.toBeginBtnJQ = $("#to-begin-btn");
            this.nextBtnJQ = $("#next-btn");
            this.toEndBtnJQ = $("#to-end-btn");
            
            this.timelineJQ = $("#timeline");
            this.timelineJQ.change( this.timelineChanged.bind(this) );

            this.input = new objects.NumberInput($("div[name=time]"), "");
            this.input.containerJQ.css("margin-top", 0);
            this.input.addEventListener("change", ()=> this.timeInputChanged() );

            this.isPlaying = false;
            this.timerId = undefined;
            this.timeStep = undefined;
            this.timeValues = undefined;
            this.currentFrameNumber = undefined;

            this.toBeginBtnJQ.click( this.setFirstFrame.bind(this) );
            this.previousBtnJQ.click( this.setPreviousFrame.bind(this) );
            this.playPauseBtnJQ.click( this.togglePlayPause.bind(this) );
            this.nextBtnJQ.click( this.setNextFrame.bind(this) );
            this.toEndBtnJQ.click( this.setLastFrame.bind(this) );
        }
        togglePlayPause() {
            try {
                if(!this.isPlaying) {
                    if(!this.timeStep)
                        throw new Error("AnimationPlayer::togglePlayPause: time step isn't specified");
                    
                    // stop when reached end
                    this.timerId = setInterval( ()=>{
                        if(!objects.noNewFrame)
                            return;

                        if(this.currentFrameNumber != this.timeValues.length - 1) {
                            this.setNextFrame();
                        }
                        else {
                            clearInterval(this.timerId);
                            this.playPauseBtnJQ
                                .removeClass("pause")
                                .addClass("play");
                            this.isPlaying = false;
                            this.timerId = undefined;
                            this.setFirstFrame();
                        }
                        objects.noNewFrame = false;
                    }, this.timeStep);
                    this.isPlaying = true;
                    this.playPauseBtnJQ
                        .removeClass("play")
                        .addClass("pause");
                }
                else {
                    clearInterval(this.timerId);
                    this.timerId = undefined;
                    this.isPlaying = false;
                    this.playPauseBtnJQ
                        .removeClass("pause")
                        .addClass("play");
                }
            } catch (error) {
                objects.showError(error.message);
            }
        }
        setNextFrame() {
            if(!this.timeValues)
                throw new Error("Animation player: time values are not specified");
            let nextFrame = this.currentFrameNumber + 1;
            if(this.currentFrameNumber == this.timeValues.length)
                this.currentFrameNumber = 0;
            this.setTimeFrame(nextFrame);
        }
        setPreviousFrame() {
            if(!this.timeValues)
                throw new Error("Animation player: time values are not specified");
            let previousFrame = this.currentFrameNumber - 1;
            if(previousFrame < 0)
                previousFrame = this.timeValues.length - 1;
            this.setTimeFrame(previousFrame);
        }
        setFirstFrame() {
            if(!this.timeValues)
                throw new Error("Animation player: time values are not specified");
            this.setTimeFrame(0);
        }
        setLastFrame() {
            if(!this.timeValues)
                throw new Error("Animation player: time values are not specified");
            let lastFrame = this.timeValues.length - 1;
            this.setTimeFrame(lastFrame);
        }
        setTimeValues(timeValues) {
            this.timeValues = timeValues;

            this.input.setDefaultValue(this.timeValues[0]);
            let range = new objects.Range(
                this.timeValues[0],
                this.timeValues[this.timeValues.length - 1],
                this.timeValues.length - 1 );
            this.timelineJQ.attr("min", range.min);
            this.timelineJQ.attr("max", range.max);
            this.timelineJQ.attr("step", range.getStep());
        }
        setTimeFrame(timeFrame) {
            if(!this.timeValues)
                throw new Error("Animation player: time values are not specified");
            if(timeFrame < 0 || timeFrame > this.timeValues.length)
                throw new Error("Animation player: tried to set invalid frame number");
            let time = this.timeValues[timeFrame];
            new objects.InputSetRequest("timeFrame", timeFrame).perform();
            this.currentFrameNumber = timeFrame;

            this.input.setValue(time);
            this.timelineJQ.val(time);
            this.dispatchEvent(new Event("change"));
        }
        getTimeFrame() {
            return this.currentFrameNumber;
        }
        setTimeStep(timeStep) {
            this.timeStep = timeStep;
        }
        setEnabledUi(enabled = true) {
            $(".player-btn").attr("disabled", !enabled);
            if(enabled)
                this.input.enable();
            else
                this.input.disable();
            this.timelineJQ.attr("disabled", !enabled);
        }

        // private
        timeInputChanged() {
            if( !this.input.isCompleted() ) {
                this.setTimeFrame(this.currentFrameNumber);
                return;
            }
            let time = this.input.getValue();
            let frame = this.frameByTime(time);
            this.setTimeFrame(frame);
        }
        timelineChanged() {
            let time = this.timelineJQ.val();
            let frame = this.frameByTime(time);
            this.setTimeFrame(frame);
        }
        frameByTime(time) {
            let t = this.timeValues.reduce((prevVal, currVal)=>{
                let d1 = Math.abs(time - prevVal);
                let d2 = Math.abs(time - currVal);
                return d1 > d2? currVal : prevVal;
            }, this.timeValues[0]);
            return this.timeValues.findIndex(val => val == t, this);
        }
    }



    class TimeService {
        constructor() {
            this.nTimeFrames = null;
            this.timeValues = null; // index - frameNumber, value - timeValue 
            this.actualFrameNumber = undefined;
            
            this.player = new AnimationPlayer();
            this.stop();
            this.player.addEventListener("change", this.playerChanged.bind(this));
        }
        run(){
            try {
                this.player.setEnabledUi(true);
                this.updateTimeData();
                this.restore();
            } catch(error) {
                objects.showError(error.message);
            }
        }
        stop(){
            this.player.setEnabledUi(false);
        }
        store() {
            let problem = $("#problem-select").val();
            if(problem == "none")
                return;

            let dataStr = localStorage.getItem(problem);
            let data;
            if(dataStr == null)
                data = {};
            else
                data = JSON.parse(dataStr);
            data.frameNumber = this.player.getTimeFrame();
            localStorage.setItem(problem, JSON.stringify(data));
        }
        restore() {
            let problem = $("#problem-select").val();
            if(problem == "none") {
                this.player.setFirstFrame();
                return;
            }
            
            let dataStr = localStorage.getItem(problem);
            if(dataStr == null) {
                this.player.setFirstFrame();
                return;
            }
            let data = JSON.parse(dataStr);
            if( !isNaN(data.frameNumber) )
                this.player.setTimeFrame(data.frameNumber);
        }
        setTimeStep(timeStep) {
            this.player.setTimeStep(timeStep);
        }

        // private
        updateTimeData() {
            this.updateTimeFramesCount();
            this.updateTimeValues();
            this.player.setTimeValues( this.timeValues );
        }
        updateTimeFramesCount() {
            let req = new objects.GetRequest("vs/time/count").perform();
            this.nTimeFrames = req.receivedData();
        }
        updateTimeValues() {
            let req = new objects.GetRequest("vs/time/timevalues", "");
            this.timeValues = req.perform().receivedData();
        }
        playerChanged() {
            this.store();
        }
    }
    objects.timeService = new TimeService();
})()