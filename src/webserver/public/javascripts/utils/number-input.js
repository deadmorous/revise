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
    class NumberInput extends EventTarget {
        constructor(containerJQ, placeholder) {
            super();
            this.containerJQ = containerJQ;
            this.fieldJQ = $("<input type=text>")
                .attr( "placeholder", placeholder )
                .attr( "name", name )
                .attr( "autocomplete", "off" )
                .attr( "size", 1 )
                .appendTo( this.containerJQ );

            this.defaultValue = undefined;

            let changeCb = ()=> this.dispatchEvent(new Event("change"));
            this.fieldJQ.change(this, changeCb.bind(this));
        }
        getValue() {
            let valStr = this.fieldJQ.val();
            if(valStr == "") 
                throw new Error("Empty input string");
            if( isNaN(+valStr) )
                throw new Error("Invalid input value");
            return +valStr;
        }
        setValue(number) {
            if( isNaN(+number) )
                throw new Error(`ERROR NumberInput::setValue: ${number} is not a number`);
            number = +number;
            if(number == 0)
                this.fieldJQ.val("0");
            else if(Math.abs(number) < 1)
                this.fieldJQ.val( number.toPrecision(2) );
            else if(Number.isInteger(number))
                this.fieldJQ.val( number.toFixed(0) );
            else if(Math.abs(number) < 10)
                this.fieldJQ.val( number.toFixed(1) );
            else
                this.fieldJQ.val( number.toFixed(0) );
        }
        disable() {
            this.fieldJQ.attr("disabled", true);
        }
        enable() {
            this.fieldJQ.attr("disabled", false);
        }
        setDefaultValue(number) {
            if( isNaN(+number) ) 
                throw new Error(`ERROR NumberInput::setDefaultValue: ${number} is not a number`);
            this.defaultValue = +number;
        }
        reset() {
            if( isNaN(this.defaultValue) )
                this.fieldJQ.val("");
            else
                this.setValue(this.defaultValue);
        }
        isCompleted() {
            let valStr = this.fieldJQ.val();
            if(valStr == "") 
                return false;
            if( isNaN(+valStr) )
                return false;
            return true;
        }
        isEmpty() {
            return this.fieldJQ.val() == "";
        }
    }
    objects.NumberInput = NumberInput;


    class InputWithSlider extends NumberInput {
        constructor(containerJQ, placeholder) {
            super(containerJQ, placeholder);
            this.containerJQ = containerJQ;

            this.sliderJQ = $("<input type='range'>")
                .attr( "form", "" )
                .appendTo( this.containerJQ )
                .hide();

            // EVENTS ESTABLISHMENT
            let focusOut = (e)=>{
                if( e.relatedTarget == this.fieldJQ[0] || e.relatedTarget == this.sliderJQ[0] )
                    return;
                this.sliderJQ.hide();
            }

            this.fieldJQ.focus( this.showSlider.bind(this) );
            this.sliderJQ.focus(this, ()=>{
                this.showSlider();
                this.setValue(this.sliderJQ.val());
                this.dispatchEvent( new Event("change") );
            });
            this.fieldJQ.focusout( focusOut.bind(this) );
            this.sliderJQ.focusout( focusOut.bind(this) );
            
            let onSliderChange = () => {
                this.setValue( this.sliderJQ.val() );
                this.dispatchEvent(new Event("change"));
            };
            this.sliderJQ[0].oninput = onSliderChange.bind(this);

            let onTextChange = () => this.sliderJQ.val( this.fieldJQ.val() );
            this.fieldJQ[0].oninput = onTextChange.bind(this);

            this.range = new objects.Range(0, 1, 100);
        }

        showSlider() {
            let fieldStr = this.fieldJQ.val();
            if( !isNaN(+fieldStr) )
                this.sliderJQ.val(+fieldStr);
            this.sliderJQ.show();
            this.positionSlider();
        }

        hideSlider() {
            this.sliderJQ.hide();
        }
        positionSlider() {
            this.sliderJQ.css( 'position', 'absolute' );

            this.sliderJQ.css({
                left: (this.fieldJQ.position().left 
                    + this.fieldJQ.width()/2
                    - this.sliderJQ.width()/2)
                    + "px",
                top: (this.fieldJQ.position().top
                    + this.fieldJQ.height() 
                    - this.sliderJQ.height()/2 + 14)
                    + "px"
            });
        }
        getRelativeValue() {
            if(this.range.max == this.range.min)
                return 0;
            let absVal = this.getValue();
            let relVal = (absVal - this.range.min) / (this.range.max - this.range.min);
            return relVal;
        }
        setRelativeValue(relValue) {
            let absVal = this.range.min * (1 - relValue) + relValue * this.range.max;
            this.setValue(absVal);
        }
        setRange(range) {
            if(this.isCompleted()) {
                let relVal = this.getRelativeValue();
                this.range = range;
                this.setRelativeValue(relVal);
            } else 
                this.range = range;

            this.sliderJQ.attr("min", range.min);
            this.sliderJQ.attr("max", range.max);
            this.sliderJQ.attr("step", range.getStep());
        }
    }
    objects.InputWithSlider = InputWithSlider;
})()