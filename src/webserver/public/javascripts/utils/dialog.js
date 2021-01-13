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
    class Dialog {
		constructor(id) {
			this.submitHandlers = [];
			this.nodeJQ = $(`#${id}`);
			this.nodeJQ.dialog({
				autoOpen: false,
				closeOnEscape: true,
				modal: true,
				closeText: "",
				width: 'auto',
				buttons: [
					{
						text: "Ok",
						click: this.submit.bind(this)
					},
					{
						text: "Cancel",
						click: this.cancel.bind(this)
					}
				],
				close: ()=>{
					this.submitHandlers = [];
					this.reset();
				},
				position: { my: "center", at: "center", of: window}
			});
			this.nodeJQ.removeClass("hidden");
		}
		show() {
			this.nodeJQ.dialog("open");
		}
		hide() {
			this.nodeJQ.dialog("close");
		}
		cancel() {
			this.submitHandlers = [];
			this.nodeJQ.dialog("close");
		}
		setTitle(title) {
			this.nodeJQ.dialog("option", "title", title);
		}
		submit() {
			try{
				this.submitHandlers.forEach(h => h.call(null));
				this.submitHandlers = [];
				this.hide();
				this.reset();
			} catch(error) {
				objects.showError(error);
			};
		}
		addSubmitHandler(handler) {
			this.submitHandlers.push(handler);
		}
	}
    objects.Dialog = Dialog;
})()